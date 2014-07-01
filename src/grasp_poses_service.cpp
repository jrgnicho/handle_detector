#include "affordances.h"
#include "handle_detector/CylinderArrayMsg.h"
#include "handle_detector/CylinderMsg.h"
#include "handle_detector/HandleListMsg.h"
#include <handle_detector/GraspPoseCandidates.h>
#include <ctype.h>
#include "cylindrical_shell.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <iostream>
#include "messages.h"
#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sstream>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <vector>
#include <godel_surface_detection/detection/surface_detection.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#define EIGEN_DONT_PARALLELIZE


// constants
const std::string SENSOR_CLOUD_TOPIC = "sensor_cloud";
const std::string FILTERED_CLOUD_TOPIC = "filtered_cloud";
const std::string GRAPS_POSE_CANDIDATES_SERVICE = "grasp_poses";
const std::string HANDLES_MARKER_ARRAY_TOPIC = "handle_markers";
const double WAIT_FOR_MESSAGE_TIMEOUT = 5.0f;

namespace tf
{
	typedef boost::shared_ptr<tf::TransformListener> TransformListenerPtr;
}

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef pcl::PointCloud<pcl::PointXYZRGBA> CloudRGBA;

typedef std::vector<CylindricalShell> CylindricalShells;
typedef std::vector< std::vector<CylindricalShell> > Handles;

class GraspPosesServer
{
public:
	GraspPosesServer()
	{

	}

	~GraspPosesServer()
	{

	}

	void run()
	{
		if(init())
		{
			ros::spin();
		}
		else
		{
			ROS_ERROR_STREAM("Grasp Pose server initialization failed, exiting");
		}
	}

protected:

	bool init()
	{
		ros::NodeHandle nh;

		// initializing service server
		target_pose_server_ = nh.advertiseService(GRAPS_POSE_CANDIDATES_SERVICE,&GraspPosesServer::grasp_pose_service_callback,this);

		// initializing publishers
		filtered_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(FILTERED_CLOUD_TOPIC,1);
		handle_makers_pub_ = nh.advertise<visualization_msgs::MarkerArray>(HANDLES_MARKER_ARRAY_TOPIC,1);

		// tf setup
		tf_listener_ptr_ =  tf::TransformListenerPtr(new tf::TransformListener(nh,ros::Duration(1.0f)));


		return load_parameters();
	}

	bool load_parameters()
	{
		ros::NodeHandle ph("~");
		affordances_.initParams(ph);

		workspace_max_.setValue(
				affordances_.workspace_limits.max_x,
				affordances_.workspace_limits.max_y,
				affordances_.workspace_limits.max_z);

		workspace_min_.setValue(
				affordances_.workspace_limits.min_x,
				affordances_.workspace_limits.min_y,
				affordances_.workspace_limits.min_z);

		geometry_msgs::Vector3 min, max;
		tf::vector3TFToMsg(workspace_min_,min);
		tf::vector3TFToMsg(workspace_max_,max);

		ROS_INFO_STREAM("Workspace bounds defined from "<<min<<" to "<<max);

		return surf_detect_.load_parameters("~/surface_detection");
	}

	bool wait_for_point_cloud_msg(sensor_msgs::PointCloud2& msg)
	{
		// grab sensor data snapshot
		ros::NodeHandle nh;
		sensor_msgs::PointCloud2ConstPtr msg_ptr =
				ros::topic::waitForMessage<sensor_msgs::PointCloud2>(SENSOR_CLOUD_TOPIC,nh,
						ros::Duration(WAIT_FOR_MESSAGE_TIMEOUT));

		// check for empty message
		if(msg_ptr != sensor_msgs::PointCloud2ConstPtr())
		{
			ROS_INFO_STREAM("grasp planner server received point cloud msg in frame "<<msg_ptr->header.frame_id);
			msg = *msg_ptr;
		}
		else
		{
			ROS_ERROR_STREAM("grasp planner server could not received cloud msg");
		}

		return msg_ptr != sensor_msgs::PointCloud2ConstPtr();
	}

	bool grasp_pose_service_callback(handle_detector::GraspPoseCandidates::Request& req,
			handle_detector::GraspPoseCandidates::Response& res)
	{
		sensor_msgs::PointCloud2 sensor_cloud_msg,filtered_cloud_msg;
		handle_detector::HandleListMsg handles_msg;
		Cloud sensor_cloud, filtered_cloud;
		tf::Transform world_to_sensor_tf;
		Eigen::Affine3d eigen3d;

		if(wait_for_point_cloud_msg(sensor_cloud_msg))
		{
			// convert ros msg to pcl cloud
			pcl::fromROSMsg(sensor_cloud_msg,sensor_cloud);

			// transform cloud to world coordinate frame
			lookup_transform(req.planning_frame_id,sensor_cloud_msg.header.frame_id,world_to_sensor_tf);
			tf::transformTFToEigen(world_to_sensor_tf,eigen3d);
			pcl::transformPointCloud(sensor_cloud,sensor_cloud,Eigen::Affine3f(eigen3d));

		}
		else
		{
			//res.succeeded = false;
			return false;
		}

		// filter workspace and find surfaces
		std::vector<Cloud::Ptr> surfaces;
		if(filter_workspace(sensor_cloud,filtered_cloud) && detect_surfaces(filtered_cloud,surfaces))
		{
			ROS_INFO_STREAM("Surfaces found: "<< surfaces.size());

			for(int i = 0; i < surfaces.size();i++)
			{
				Cloud &surface = *surfaces[i];
				handle_detector::HandleListMsg h;
				if(detect_handles(surface,h))
				{
					handles_msg.handles.insert(handles_msg.handles.end(),h.handles.begin(),h.handles.end());
					ROS_INFO_STREAM("Found "<<h.handles.size()<<" handles from surface "<<i);
				}

			}

		}
		else
		{
			ROS_ERROR_STREAM("Surfaces were not found, exiting");
			return false;
		}



		if( !handles_msg.handles.empty() )
		{
			ROS_INFO_STREAM("Processing "<<handles_msg.handles.size()<<" candidate handles");

			geometry_msgs::Pose grasp_pose,cylinder_pose;
			tf::Transform grasp_tf, cylinder_tf;
			tf::Vector3 normal,axis;
			double angle = 2*M_PI/req.candidates_per_pose;

			bool found = false;
			std::vector<handle_detector::CylinderArrayMsg> &handles = handles_msg.handles;
			for(int i = 0;i < handles.size() && !found; i++)
			{
				handle_detector::CylinderArrayMsg &ca = handles[i];
				std::vector<handle_detector::CylinderMsg> &c = ca.cylinders;

				for(int j = 0; j < ca.cylinders.size();j++)
				{
					handle_detector::CylinderMsg &cylinder = ca.cylinders[j];

					if(cylinder.radius*2 < req.gripper_workrange )
					{

						ROS_INFO_STREAM("Processing handle "<<i<<" with pose "<<cylinder.pose);

						// converting msg to tf
						tf::poseMsgToTF(cylinder.pose,cylinder_tf);
						tf::vector3MsgToTF(cylinder.normal,normal);
						tf::vector3MsgToTF(cylinder.axis,axis);

						tf::Vector3 rz_grasp = normal.normalized();
						tf::Vector3 rx_grasp = axis.normalized();
						tf::Vector3 ry_grasp = (rz_grasp.cross(rx_grasp)).normalized();

/*						tf::Matrix3x3 rot_grasp = tf::Matrix3x3(rx_grasp.getX(),rx_grasp.getY(),rx_grasp.getZ(),
								ry_grasp.getX(),ry_grasp.getY(),ry_grasp.getZ(),
								rz_grasp.getX(),rx_grasp.getY(),rz_grasp.getZ());*/
						tf::Quaternion rot_grasp = cylinder_tf.getRotation()*tf::Quaternion(tf::Vector3(1,0,0),M_PI/2.0f);

						// creating multiple candidate poses
						geometry_msgs::PoseArray grasp_poses;
						for(int e = -2; e < req.candidates_per_pose-2;e++)
						{
							// rotating about cylinder axis by the angle
							//tf::Quaternion rot_about_axis = tf::Quaternion(rx_grasp,e*angle);
							//grasp_tf.setBasis(rot_grasp * tf::Matrix3x3(rot_about_axis));
							tf::Quaternion rot_about_axis = tf::Quaternion(tf::Vector3(0,1,0),e*angle);
							grasp_tf.setRotation(rot_grasp * rot_about_axis);

							// setting position
							//tf::Vector3 grasp_pos = cylinder_tf*tf::Vector3(0,0.,cylinder.extent);
							grasp_tf.setOrigin(cylinder_tf.getOrigin());

							tf::poseTFToMsg(grasp_tf,grasp_pose);
							grasp_poses.poses.push_back(grasp_pose);
						}
						res.candidate_grasp_poses.push_back(grasp_poses);
						ROS_INFO_STREAM("Grasp pose 0:\n"<<grasp_poses.poses[0]);

						// filter cylinder from point cloud and publishing
						filter_cylinder(cylinder,filtered_cloud,filtered_cloud);
						pcl::toROSMsg(filtered_cloud,filtered_cloud_msg);
						filtered_cloud_msg.header.frame_id = req.planning_frame_id;
						filtered_cloud_pub_.publish(filtered_cloud_msg);

						// creating cylinder marker
						std_msgs::ColorRGBA color;
						color.r=0;
						color.g = 1;
						color.b = 1;
						color.a = 0.5f;
						visualization_msgs::Marker marker;
						marker.header.frame_id = req.planning_frame_id;
						marker.type = marker.CYLINDER;
						marker.action = marker.ADD;
						marker.pose = cylinder.pose;
						marker.ns = "handles";
						marker.scale.x = marker.scale.y = cylinder.radius*2;
						marker.scale.z = cylinder.extent;
						marker.color = color;
						res.candidate_objects.markers.push_back(marker);

						found = true;
						break;

					}
				}

			}

			if(!res.candidate_grasp_poses.empty())
			{
				ROS_INFO_STREAM("Found "<<res.candidate_grasp_poses.size()<<" graspable handles");
				handle_makers_pub_.publish(res.candidate_objects);
			}
			else
			{
				ROS_ERROR_STREAM("Found 0 graspable handles");
			}

		}
		else
		{
			//res.succeeded = false;
			return false;
		}

		return !res.candidate_grasp_poses.empty();
	}

	bool lookup_transform(std::string planning_frame_id,std::string source_frame_id,tf::Transform &t)
	{
		// find transform of handles from tf
		tf::StampedTransform robot_to_target_tf;
		ros::Time query_time = ros::Time::now();

		if(planning_frame_id.compare(source_frame_id) == 0 || source_frame_id.empty() || planning_frame_id.empty())
		{

			t.setIdentity();
			return true;
		}
		else
		{

			if(!tf_listener_ptr_->waitForTransform(planning_frame_id,source_frame_id,query_time,ros::Duration(5)))
			{
				return false;
			}
		}

		try
		{
			tf_listener_ptr_->lookupTransform(planning_frame_id,source_frame_id,ros::Time(0),robot_to_target_tf);
			t.setOrigin(robot_to_target_tf.getOrigin());
			t.setRotation(robot_to_target_tf.getRotation());
			ROS_INFO_STREAM("Found transform from "<<planning_frame_id<<" to "<<source_frame_id);
		}
		catch(tf::TransformException &e)
		{
			ROS_ERROR_STREAM("Transform lookup failed");
			return false;
		}
		catch(tf::LookupException &e)
		{
			ROS_ERROR_STREAM("Transform lookup failed");
			return false;
		}

		return true;
	}

	bool detect_handles(const Cloud& cloud, handle_detector::HandleListMsg& handles_msg )
	{
		ROS_INFO_STREAM("Handle detection started");

		Messages message_creator;
		CylindricalShells shells = affordances_.searchAffordances(cloud.makeShared());
		Handles handles = affordances_.searchHandles(cloud.makeShared(),shells);

		handle_detector::HandleListMsg msg = message_creator.createHandleList(handles,"");
		handles_msg.handles = msg.handles;

		ROS_INFO_STREAM("Handle detection completed with "<<handles_msg.handles.size()<<" handles");

		return !handles_msg.handles.empty();
	}

	bool detect_surfaces(Cloud& in,std::vector<Cloud::Ptr> &surfaces)
	{
		surf_detect_.clear_results();
		surf_detect_.add_cloud(in);
		if(surf_detect_.find_surfaces())
		{
			surfaces = surf_detect_.get_surface_clouds();
		}

		// printing results
		std::stringstream ss;
		for(int i = 0;i< surfaces.size();i++)
		{
			ss<<"Surface "<<i<<" with "<<surfaces[i]->size()<<" points\n";
		}
		ROS_INFO_STREAM_COND(!surfaces.empty(),"Surfaces details:\n"<<ss.str());

		return !surfaces.empty();
	}

	bool filter_workspace(const Cloud &sensor_cloud,Cloud &filtered_cloud)
	{
		Cloud temp;
		pcl::copyPointCloud(sensor_cloud,temp);

		// filtering workspace bounds
		pcl::PassThrough<pcl::PointXYZ> filter;

		// remove z limits
		filter.setInputCloud(temp.makeShared());
		filter.setFilterFieldName("z");
		filter.setFilterLimits(workspace_min_.getZ(),workspace_max_.getZ());
		filter.filter(temp);

		// remove y limits
		filter.setInputCloud(temp.makeShared());
		filter.setFilterFieldName("y");
		filter.setFilterLimits(workspace_min_.getY(),workspace_max_.getY());
		filter.filter(temp);

		// remove x limits
		filter.setInputCloud(temp.makeShared());
		filter.setFilterFieldName("x");
		filter.setFilterLimits(workspace_min_.getX(),workspace_max_.getX());
		filter.filter(filtered_cloud);


		return !filtered_cloud.empty();
	}

	bool filter_cylinder(const handle_detector::CylinderMsg& cylinder_msg,const Cloud &in, Cloud &filtered)
	{
		// creating cylinder cloud
		int num_points = 20;
		double radius = 1.5f*cylinder_msg.radius;
		double delta_rot = 2*M_PI/(num_points-1);
		Cloud circle;
		for(int i = 0;i < num_points;i++)
		{
			pcl::PointXYZ p;
			p.x = radius*std::cos(i*delta_rot);
			p.y = radius*std::sin(i*delta_rot);
			p.z = 0;
			circle.push_back(p);
		}

		// transforming circle to cylinder pose in world coordinates
		tf::Transform cylinder_tf;
		Eigen::Affine3d cylinder_eig;
		tf::poseMsgToEigen(cylinder_msg.pose,cylinder_eig);
		pcl::transformPointCloud(circle,circle,Eigen::Affine3f(cylinder_eig));

		// extract points
		pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

		prism.setInputCloud(in.makeShared());
		prism.setInputPlanarHull(circle.makeShared());
		prism.setHeightLimits(-cylinder_msg.extent,cylinder_msg.extent);
		prism.setViewPoint(0,0,10);
		prism.segment(*inliers);

		// extracting remaining points
		extract.setInputCloud(in.makeShared());
		extract.setIndices(inliers);
		extract.setNegative(true);
		extract.filter(filtered);

		return !inliers->indices.empty();
	}


protected:

	// members
	std::string world_frame_id_;
	sensor_msgs::PointCloud2 filtered_cloud_msg_;
	sensor_msgs::PointCloud2 sensor_cloud_msg_;

	// ros comm
	ros::ServiceServer target_pose_server_;
	ros::Publisher filtered_cloud_pub_;
	ros::Publisher handle_makers_pub_;

	// tf
	tf::TransformListenerPtr tf_listener_ptr_;

	// parameters
	tf::Vector3 workspace_min_;
	tf::Vector3 workspace_max_;

	// handle detection
	Affordances affordances_;

	// surface detection
	godel_surface_detection::detection::SurfaceDetection surf_detect_;
};

int main(int argc,char** argv)
{
	ros::init(argc,argv,"grasp_poses_service");
	GraspPosesServer s;
	s.run();

	return 0;
}
