#include "affordances.h"
#include "handle_detector/CylinderArrayMsg.h"
#include "handle_detector/CylinderMsg.h"
#include "handle_detector/HandleListMsg.h"
#include <handle_detector/GraspPoseCandidates.h>
#include "cylindrical_shell.h"
#include "Eigen/Dense"
#include "Eigen/Core"
#include "messages.h"
#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/distances.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
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
		int attempts = 20;
		bool found = false;
		while(ros::ok() && attempts-- > 0)
		{
			sensor_msgs::PointCloud2ConstPtr msg_ptr =
					ros::topic::waitForMessage<sensor_msgs::PointCloud2>(SENSOR_CLOUD_TOPIC,nh,
							ros::Duration(WAIT_FOR_MESSAGE_TIMEOUT));

			// check for empty message and time stamp
			if(msg_ptr != sensor_msgs::PointCloud2ConstPtr() &&
					(ros::Time::now() - msg_ptr->header.stamp < ros::Duration(4)))
			{
				ROS_INFO_STREAM("grasp planner server received point cloud msg in frame "<<msg_ptr->header.frame_id);
				msg = *msg_ptr;
				found = true;
				break;
			}
			else
			{
				ROS_ERROR_STREAM("grasp planner server could not received cloud msg");
			}
		}

		return found;
	}

	bool grasp_pose_service_callback(handle_detector::GraspPoseCandidates::Request& req,
			handle_detector::GraspPoseCandidates::Response& res)
	{
		sensor_msgs::PointCloud2 sensor_cloud_msg,obstacle_cloud_msg;
		handle_detector::HandleListMsg handles_msg;
		Cloud sensor_cloud, workspace_cloud,obstacle_cloud, handle_cloud, table;
		tf::Transform world_to_sensor_tf;
		std::vector<Cloud::Ptr> surfaces;
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

		// removing workspace and table
		if(filter_workspace(sensor_cloud,workspace_cloud))
		{

			ROS_INFO_STREAM("Workspace bounds removal completed, "<< surfaces.size()<<" points remain");

		}
		else
		{
			ROS_ERROR_STREAM("Workspace bounds removal returned with no points, exiting");
			return false;
		}

		// removing table
		if(filter_tabletop(workspace_cloud,workspace_cloud,table))
		{
			obstacle_cloud += table;
			ROS_INFO_STREAM("Tabletop successfully removed, points in table: "<<table.size());
		}
		else
		{
			ROS_WARN_STREAM("Tabletop removal found no predominant surface");
		}

		// filter workspace and find surfaces
		if(detect_surfaces(workspace_cloud,surfaces))
		{
			ROS_INFO_STREAM("Surfaces found: "<< surfaces.size());

		}
		else
		{
			ROS_ERROR_STREAM("Surfaces were not found, exiting");
			return false;
		}


		// finding grasp candidates
		geometry_msgs::PoseArray poses;
		visualization_msgs::Marker marker;
		moveit_msgs::CollisionObject col;
		handle_detector::CylinderMsg cylinder;
		int selected_index = -1;
		for(int i = 0; i < surfaces.size();i++)
		{
			Cloud &surface = *surfaces[i];
			handle_detector::HandleListMsg h;

			// finding handles
			if(detect_handles(surface,h))
			{
				ROS_INFO_STREAM("Found "<<h.handles.size()<<" handles from surface "<<i);

			}
			else
			{
				continue;
			}

			// finding candidate poses
			poses.poses.clear();
			if(find_candidate_poses(h,surface,req,poses,cylinder))
			{
				ROS_INFO_STREAM("Found "<<poses.poses.size()<<" candidate poses for surface "<<i);
				res.candidate_grasp_poses.push_back(poses);
			}
			else
			{
				continue;
			}

			// creating marker
			create_marker(cylinder,marker);
			marker.header.frame_id = req.planning_frame_id;
			res.candidate_objects.markers.push_back(marker);

			// creating collision object
			std::stringstream ss;
			ss<<"cylinder"<<i;
			create_collision_obj(cylinder,col);
			col.header.frame_id = req.planning_frame_id;
			col.id = ss.str();
			res.candidate_collision_objects.push_back(col);
			selected_index = i;

			// removing handle from surface
			filter_cylinder(cylinder,surface,surface,handle_cloud);
			break;

		}

		// printing results
		if(!res.candidate_grasp_poses.empty())
		{
			ROS_INFO_STREAM("Adding "<<surfaces.size()<<" obstacles to obstacle point cloud");

			for(int i = 0;i < surfaces.size();i++)
			{
				obstacle_cloud+=*surfaces[i];
			}

			obstacle_cloud.header.frame_id = req.planning_frame_id;
			pcl::toROSMsg(obstacle_cloud,obstacle_cloud_msg);
			filtered_cloud_pub_.publish(obstacle_cloud_msg);

			ROS_INFO_STREAM("Found "<<res.candidate_grasp_poses.size()<<" graspable handles");
			handle_makers_pub_.publish(res.candidate_objects);
		}
		else
		{
			ROS_ERROR_STREAM("Found 0 graspable handles");
		}

		return !res.candidate_grasp_poses.empty();
	}

	bool find_candidate_poses(const handle_detector::HandleListMsg& handle_msg,const Cloud& cluster,
			const handle_detector::GraspPoseCandidates::Request& req,geometry_msgs::PoseArray &candidate_poses,
			handle_detector::CylinderMsg &selected_cylinder)
	{
		geometry_msgs::Pose grasp_pose,cylinder_pose;
		tf::Transform grasp_tf, cylinder_tf;
		tf::Vector3 normal,axis;
		double angle = 2*M_PI/req.candidates_per_pose;
		bool found = false;

		// collecting all cylinders
		std::vector<handle_detector::CylinderMsg> c;
		for(int i =0;i < handle_msg.handles.size();i++)
		{
			const handle_detector::CylinderArrayMsg& c_temp = handle_msg.handles[i];
			c.insert(c.end(),c_temp.cylinders.begin(),c_temp.cylinders.end());
		}

		for(int j = 0; j < c.size();j++)
		{
			handle_detector::CylinderMsg &cylinder = c[j];

			if(cylinder.radius*2 < req.gripper_workrange )
			{
				ROS_INFO_STREAM("Processing handle "<<j<<" with pose "<<cylinder.pose);

				// filtering points in handle
				Cloud filtered, handle;
				cylinder.extent = 2.0f;
				filter_cylinder(cylinder,cluster,filtered,handle);

				// fitting candidate cylinder to data
				fit_cylinder_to_cluster(cylinder,handle,cylinder);

				tf::poseMsgToTF(cylinder.pose,cylinder_tf);

				// rotating about local x axis to create target pose (local z vector pointing away)
				tf::Quaternion rot_grasp = cylinder_tf.getRotation()*tf::Quaternion(tf::Vector3(1,0,0),M_PI/2.0f);

				// creating multiple candidate poses
				for(int e = -2; e < req.candidates_per_pose-2;e++)
				{
					// rotating about cylinder axis in order to change the direction of local z vector
					tf::Quaternion rot_about_axis = tf::Quaternion(tf::Vector3(0,1,0),e*angle);
					grasp_tf.setRotation(rot_grasp * rot_about_axis);

					// setting position
					grasp_tf.setOrigin(cylinder_tf.getOrigin());

					tf::poseTFToMsg(grasp_tf,grasp_pose);
					candidate_poses.poses.push_back(grasp_pose);
				}

				// copying cylinder
				selected_cylinder = cylinder;

				found = true;
				break;

			}
		}

		return found;
	}

	void create_marker(const handle_detector::CylinderMsg& cylinder, visualization_msgs::Marker &marker)
	{
		// creating cylinder marker
		std_msgs::ColorRGBA color;
		color.r=0;
		color.g = 1;
		color.b = 1;
		color.a = 0.5f;
		marker.type = marker.CYLINDER;
		marker.action = marker.ADD;
		marker.pose = cylinder.pose;
		marker.ns = "handles";
		marker.scale.x = marker.scale.y = cylinder.radius*2;
		marker.scale.z = cylinder.extent;
		marker.color = color;
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

	bool filter_tabletop(const Cloud &in,Cloud &filtered,Cloud &table)
	{
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
		pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_ptr(
				new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(in.makeShared()));
		pcl::RandomSampleConsensus<pcl::PointXYZ> sac (model_ptr, 0.02f);

		if(sac.computeModel())
		{
			sac.getInliers(inliers->indices);
			extract.setNegative(false);

			// extracting
			extract.setInputCloud(in.makeShared());
			extract.setIndices(inliers);
			extract.setNegative(true);
			extract.filter(filtered);
			extract.setNegative(false);
			extract.filter(table);

		}

		if(table.empty())
		{
			return false;
		}
		else
		{

			// finding centroid
			Eigen::Vector4d centroid;
			pcl::compute3DCentroid(table,centroid);

			// filtering points below table
			Cloud below;
			pcl::PassThrough<pcl::PointXYZ> filter;
			filter.setInputCloud(filtered.makeShared());
			filter.setFilterFieldName("z");
			filter.setFilterLimits(workspace_min_.getZ(),centroid[2]);
			filter.setNegative(true);
			filter.filter(filtered);
			filter.setNegative(false);
			filter.filter(below);

			// adding extra points to table
			table += below;
		}

		return !table.empty();
	}

	bool filter_cylinder(const handle_detector::CylinderMsg& cylinder_msg,const Cloud &source, Cloud &filtered,Cloud& cylinder)
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

		// copying point cloud
		Cloud in;
		pcl::copyPointCloud(source,in);

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
		prism.setHeightLimits(-0.5f*cylinder_msg.extent,0.5f*cylinder_msg.extent);
		prism.setViewPoint(0,0,10);
		prism.segment(*inliers);

		// extracting remaining points
		extract.setInputCloud(in.makeShared());
		extract.setIndices(inliers);
		extract.setNegative(true);
		extract.filter(filtered);
		extract.setNegative(false);
		extract.filter(cylinder);
		return !inliers->indices.empty();
	}

	bool fit_cylinder_to_cluster(const handle_detector::CylinderMsg& cylinder_estimate, const Cloud& cluster,
			handle_detector::CylinderMsg &cylinder_fitted)
	{
		// finding extend
		pcl::PointXYZ min,max;
		handle_detector::CylinderMsg cylinder_refined;
		cylinder_refined = cylinder_estimate;
		cylinder_refined.extent = pcl::getMaxSegment(cluster,min,max);// plus 20%

		// preparing cylinder info
		tf::Vector3 axis;
		tf::Vector3 pos;
		pcl::ModelCoefficients::Ptr coeff_ptr(new pcl::ModelCoefficients());
		tf::vector3MsgToTF(cylinder_estimate.axis,axis);
		tf::pointMsgToTF(cylinder_estimate.pose.position,pos);
		axis.normalize();

		// filing cylinder axis coefficients
		coeff_ptr->values.resize(6);
		coeff_ptr->values[0] = pos.getX();coeff_ptr->values[1] = pos.getY();coeff_ptr->values[2] = pos.getZ();
		coeff_ptr->values[3] = axis.getX();coeff_ptr->values[4] = axis.getY();coeff_ptr->values[5] = axis.getZ();

		Cloud extends;
		Cloud projected;
		extends.push_back(min);
		extends.push_back(max);

		// projecting poinst onto line
		pcl::ProjectInliers<pcl::PointXYZ> proj;
		proj.setModelType(pcl::SACMODEL_LINE);
		proj.setInputCloud(extends.makeShared());
		proj.setModelCoefficients(coeff_ptr);
		proj.filter(projected);

		if(!projected.empty())
		{
			// modifing cylinder position so that it is in the middle of the projected min and max
			pcl::PointXYZ new_min = projected.points[0];
			tf::Vector3 min_proj = tf::Vector3(new_min.x,new_min.y,new_min.z);
			axis = (pos-min_proj).normalized();
			tf::Vector3 new_pos = min_proj + cylinder_refined.extent*0.5f*axis;
			tf::pointTFToMsg(new_pos,cylinder_refined.pose.position);

			ROS_INFO_STREAM("Cylinder position changed from:\n"<<cylinder_estimate.pose.position<<
					"\n to: \n"<<cylinder_refined.pose.position);
		}

		cylinder_fitted = cylinder_refined;


		return !projected.empty();
	}

	void create_collision_obj(const handle_detector::CylinderMsg& c,moveit_msgs::CollisionObject &obj)
	{
		// creating shape
		shape_msgs::SolidPrimitive shape;
		shape.type = shape.CYLINDER;
	    shape.dimensions.resize(2);
	    shape.dimensions[shape.CYLINDER_HEIGHT] = c.extent;
	    shape.dimensions[shape.CYLINDER_RADIUS] = c.radius;

	    obj.primitives.push_back(shape);
	    obj.primitive_poses.push_back(c.pose);
	    obj.operation = obj.ADD;
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
