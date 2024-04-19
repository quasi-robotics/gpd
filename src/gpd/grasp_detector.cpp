#include <gpd/grasp_detector.h>

namespace gpd {

GraspDetector::GraspDetector(const std::string &config_filename) {
  Eigen::initParallel();

  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();

  // Read hand geometry parameters.
  std::string hand_geometry_filename =
      config_file.getValueOfKeyAsString("hand_geometry_filename", "");
  if (hand_geometry_filename == "0") {
    hand_geometry_filename = config_filename;
  }

  GraspDetectionParameters param;
  // Read plotting parameters.
  param.plot_normals_ = config_file.getValueOfKey<bool>("plot_normals", false);
  param.plot_samples_ = config_file.getValueOfKey<bool>("plot_samples", true);
  param.plot_candidates_ = config_file.getValueOfKey<bool>("plot_candidates", false);
  param.plot_filtered_candidates_ =
      config_file.getValueOfKey<bool>("plot_filtered_candidates", false);
  param.plot_valid_grasps_ =
      config_file.getValueOfKey<bool>("plot_valid_grasps", false);
  param.plot_clustered_grasps_ =
      config_file.getValueOfKey<bool>("plot_clustered_grasps", false);
  param.plot_selected_grasps_ =
      config_file.getValueOfKey<bool>("plot_selected_grasps", false);

  // Create object to generate grasp candidates.
  candidate::CandidatesGenerator::Parameters generator_params;
  param.generator_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  param.generator_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  param.generator_params.remove_statistical_outliers_ =
      config_file.getValueOfKey<bool>("remove_outliers", false);
  param.generator_params.sample_above_plane_ =
      config_file.getValueOfKey<bool>("sample_above_plane", false);
  param.generator_params.voxelize_ =
      config_file.getValueOfKey<bool>("voxelize", true);
  param.generator_params.voxel_size_ =
      config_file.getValueOfKey<double>("voxel_size", 0.003);
  param.generator_params.normals_radius_ =
      config_file.getValueOfKey<double>("normals_radius", 0.03);
  param.generator_params.refine_normals_k_ =
      config_file.getValueOfKey<int>("refine_normals_k", 0);
  param.generator_params.workspace_ =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");

  param.hand_search_params.hand_geometry_ = candidate::HandGeometry(hand_geometry_filename);
  param.hand_search_params.nn_radius_frames_ =
      config_file.getValueOfKey<double>("nn_radius", 0.01);
  param.hand_search_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  param.hand_search_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  param.hand_search_params.num_orientations_ =
      config_file.getValueOfKey<int>("num_orientations", 8);
  param.hand_search_params.num_finger_placements_ =
      config_file.getValueOfKey<int>("num_finger_placements", 10);
  param.hand_search_params.deepen_hand_ =
      config_file.getValueOfKey<bool>("deepen_hand", true);
  param.hand_search_params.hand_axes_ =
      config_file.getValueOfKeyAsStdVectorInt("hand_axes", "2");
  param.hand_search_params.friction_coeff_ =
      config_file.getValueOfKey<double>("friction_coeff", 20.0);
  param.hand_search_params.min_viable_ =
      config_file.getValueOfKey<int>("min_viable", 6);

  // Read grasp image parameters.
  std::string image_geometry_filename =
      config_file.getValueOfKeyAsString("image_geometry_filename", "");
  if (image_geometry_filename == "0") {
    image_geometry_filename = config_filename;
  }
  param.image_params = descriptor::ImageGeometry(image_geometry_filename);

  param.model_file_ = config_file.getValueOfKeyAsString("model_file", "");
  param.weights_file_ = config_file.getValueOfKeyAsString("weights_file", "");
  
  if (!param.model_file_.empty() || !param.weights_file_.empty()) {
    param.device_ = config_file.getValueOfKey<int>("device", 0);
    param.batch_size_ = config_file.getValueOfKey<int>("batch_size", 100);
  }
  
  // Read additional grasp image creation parameters.
  param.remove_plane_ =  config_file.getValueOfKey<bool>(
      "remove_plane_before_image_calculation", false);

  // Read grasp filtering parameters based on robot workspace and gripper width.
  param.workspace_grasps_ = config_file.getValueOfKeyAsStdVectorDouble(
      "workspace_grasps", "-1 1 -1 1 -1 1");

  param.min_aperture_ = config_file.getValueOfKey<double>("min_aperture", 0.0);
  param.max_aperture_ = config_file.getValueOfKey<double>("max_aperture", 0.085);
  
  // Read grasp filtering parameters based on approach direction.
  param.filter_approach_direction_ =
      config_file.getValueOfKey<bool>("filter_approach_direction", false);
  param.direction_ = config_file.getValueOfKeyAsStdVectorDouble("direction", "1 0 0");
  param.thresh_rad_ = config_file.getValueOfKey<double>("thresh_rad", 2.3);
  
  // Read clustering parameters.
  param.min_inliers_ = config_file.getValueOfKey<int>("min_inliers", 1);

  // Read grasp selection parameters.
  param.num_selected_ = config_file.getValueOfKey<int>("num_selected", 100);

  init(param);
}

GraspDetector::GraspDetector(GraspDetectionParameters& param) {
  Eigen::initParallel();
  init(param);
}

void GraspDetector::init(GraspDetectionParameters& param)
{
    std::cout << param.hand_search_params.hand_geometry_;

    //TODO merge with GraspDetector(const std::string &config_filename)
    //Read plotting parameters.
    plot_normals_ = param.plot_normals_;
    plot_samples_ = param.plot_samples_;
    plot_candidates_ = param.plot_candidates_;
    plot_filtered_candidates_ = param.plot_filtered_grasps_;
    plot_valid_grasps_ = param.plot_valid_grasps_;
    plot_clustered_grasps_ = param.plot_clusters_;
    plot_selected_grasps_ = param.plot_selected_grasps_;
    printf("============ PLOTTING ========================\n");
    printf("plot_normals: %s\n", plot_normals_ ? "true" : "false");
    printf("plot_samples %s\n", plot_samples_ ? "true" : "false");
    printf("plot_candidates: %s\n", plot_candidates_ ? "true" : "false");
    printf("plot_filtered_candidates: %s\n",
           plot_filtered_candidates_ ? "true" : "false");
    printf("plot_valid_grasps: %s\n", plot_valid_grasps_ ? "true" : "false");
    printf("plot_clustered_grasps: %s\n",
           plot_clustered_grasps_ ? "true" : "false");
    printf("plot_selected_grasps: %s\n",
           plot_selected_grasps_ ? "true" : "false");
    printf("==============================================\n");

    // // Create object to generate grasp candidates.
    candidates_generator_ = std::make_unique<candidate::CandidatesGenerator>(
            param.generator_params, param.hand_search_params);

    printf("============ CLOUD PREPROCESSING =============\n");
    printf("voxelize: %s\n", param.generator_params.voxelize_ ? "true" : "false");
    printf("voxel_size: %.3f\n", param.generator_params.voxel_size_);
    printf("remove_outliers: %s\n",
           param.generator_params.remove_statistical_outliers_ ? "true" : "false");
    printStdVector(param.generator_params.workspace_, "workspace");
    printf("sample_above_plane: %s\n",
           param.generator_params.sample_above_plane_ ? "true" : "false");
    printf("normals_radius: %.3f\n", param.generator_params.normals_radius_);
    printf("refine_normals_k: %d\n", param.generator_params.refine_normals_k_);
    printf("==============================================\n");

    printf("============ CANDIDATE GENERATION ============\n");
    printf("num_samples: %d\n", param.hand_search_params.num_samples_);
    printf("num_threads: %d\n", param.hand_search_params.num_threads_);
    printf("nn_radius: %3.2f\n", param.hand_search_params.nn_radius_frames_);
    printStdVector(param.hand_search_params.hand_axes_, "hand axes");
    printf("num_orientations: %d\n", param.hand_search_params.num_orientations_);
    printf("num_finger_placements: %d\n",
           param.hand_search_params.num_finger_placements_);
    printf("deepen_hand: %s\n",
           param.hand_search_params.deepen_hand_ ? "true" : "false");
    printf("friction_coeff: %3.2f\n", param.hand_search_params.friction_coeff_);
    printf("min_viable: %d\n", param.hand_search_params.min_viable_);
    printf("==============================================\n");

    std::cout << param.image_params;
 
     classifier_ = net::Classifier::create(
                param.model_file_, param.weights_file_, static_cast<net::Classifier::Device>(param.device_),
                param.batch_size_);
    min_score_ = param.min_score_diff_;
    printf("============ CLASSIFIER ======================\n");
    printf("model_file: %s\n", param.model_file_.c_str());
    printf("weights_file: %s\n", param.weights_file_.c_str());
    printf("batch_size: %d\n", param.batch_size_);
    printf("min_score_: %f\n", min_score_);
    printf("==============================================\n");
    printf("thresh_rad_: %3.4f\n", param.thresh_rad_);
    printf("device_: %d\n", param.device_);

    //Create object to create grasp images from grasp candidates (used for classification).
    image_generator_ = std::make_unique<descriptor::ImageGenerator>(
            param.image_params, param.hand_search_params.num_threads_,
            param.hand_search_params.num_orientations_, false, param.remove_plane_);

    // Read grasp filtering parameters based on robot workspace and gripper width.
    workspace_grasps_ = param.workspace_grasps_;
    min_aperture_ = param.min_aperture_;
    max_aperture_ = param.max_aperture_;

    printf("============ CANDIDATE FILTERING =============\n");
    printStdVector(workspace_grasps_, "candidate_workspace");
    printf("min_aperture: %3.4f\n", min_aperture_);
    printf("max_aperture: %3.4f\n", max_aperture_);
    printf("==============================================\n");

    // // Read grasp filtering parameters based on approach direction.
    filter_approach_direction_ = param.filter_approach_direction_;
    direction_ << param.direction_[0], param.direction_[1], param.direction_[2];
    thresh_rad_ = param.thresh_rad_;

    printf("filter_approach_direction_ = %d\n", filter_approach_direction_);
    printf("thresh_rad_ = %f\n", thresh_rad_);
    for(int i=0; i<direction_.size(); ++i)
    {
        printf("direction = %f\n", direction_[i]);
    }

    // Read clustering parameters.
    clustering_ = std::make_unique<Clustering>(param.min_inliers_);
    cluster_grasps_ = param.min_inliers_ > 0;
    printf("============ CLUSTERING ======================\n");
    printf("min_inliers: %d\n", param.min_inliers_);
    printf("==============================================\n\n");
    // // Read grasp selection parameters.
    num_selected_ = param.num_selected_;

    // // Create plotter.
    plotter_ = std::make_unique<util::Plot>(param.hand_search_params.hand_axes_.size(),
                                            param.hand_search_params.num_orientations_);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::detectGrasps(
    const util::Cloud &cloud) {
  double t0_total = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  const candidate::HandGeometry &hand_geom = candidates_generator_->getHandSearchParams().hand_geometry_;

  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    return hands_out;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  double t0_candidates = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    return hands_out;
  }
  double t_candidates = omp_get_wtime() - t0_candidates;
  if (plot_candidates_) {
    plotter_->plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                            "Grasp candidates", hand_geom);
  }

  // 2. Filter the candidates.
  double t0_filter = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }
  double t_filter = omp_get_wtime() - t0_filter;
  if (hand_set_list_filtered.size() == 0) {
    return hands_out;
  }

  // 3. Create grasp descriptors (images).
  double t0_images = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images, hands);
  double t_images = omp_get_wtime() - t0_images;

  // 4. Classify the grasp candidates.
  double t0_classify = omp_get_wtime();
  std::vector<float> scores = classifier_->classifyImages(images);
  for (int i = 0; i < hands.size(); i++) {
    hands[i]->setScore(scores[i]);
  }
  double t_classify = omp_get_wtime() - t0_classify;

  // 5. Select the <num_selected> highest scoring grasps.
  double t0_select = omp_get_wtime();
  hands = selectGrasps(hands);
  if (plot_valid_grasps_) {
    plotter_->plotFingers3D(hands, cloud.getCloudOriginal(), "Valid Grasps",
                            hand_geom);
  }
  double t_select = omp_get_wtime() - t0_select;

  // 6. Cluster the grasps.
  double t0_cluster = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> clusters;
  if (cluster_grasps_) {
    clusters = clustering_->findClusters(hands);
    printf("Found %d clusters.\n", (int)clusters.size());
    if (clusters.size() <= 3) {
      printf(
          "Not enough clusters found! Adding all grasps from previous step.");
      for (int i = 0; i < hands.size(); i++) {
        clusters.push_back(std::move(hands[i]));
      }
    }
    if (plot_clustered_grasps_) {
      plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                              "Clustered Grasps", hand_geom);
    }
  } else {
    clusters = std::move(hands);
  }
  double t_cluster = omp_get_wtime() - t0_cluster;

  // 7. Sort grasps by their score.
  std::sort(clusters.begin(), clusters.end(), isScoreGreater);
  printf("======== Selected grasps ========\n");
  for (int i = 0; i < clusters.size(); i++) {
    printf("Grasp %d, score: %f, approach: [%f, %f, %f], position: [%f, %f, %f]\n", i, clusters[i]->getScore(),
    clusters[i]->getApproach()[0], clusters[i]->getApproach()[1], clusters[i]->getApproach()[2],
    clusters[i]->getGraspTop()[0], clusters[i]->getGraspTop()[1], clusters[i]->getGraspTop()[2]);
  }
  printf("Selected the %d best grasps.\n", (int)clusters.size());
  double t_total = omp_get_wtime() - t0_total;

  printf("======== RUNTIMES ========\n");
  printf(" - Candidate generation: %3.4fs\n", t_candidates);
  printf(" - Descriptor extraction: %3.4fs\n", t_images);
  printf(" - Selection: %3.4fs\n", t_select);
  printf(" - Classification: %3.4fs\n", t_classify);
  printf(" - Filtering: %3.4fs\n", t_filter);
  printf(" - Clustering: %3.4fs\n", t_cluster);
  printf("==========\n");
  printf(" TOTAL: %3.4fs\n", t_total);

  if (plot_selected_grasps_) {
    plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                            "Selected Grasps", hand_geom, false);
  }

  return clusters;
}

void GraspDetector::preprocessPointCloud(util::Cloud &cloud) {
  candidates_generator_->preprocessPointCloud(cloud);
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsWorkspace(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const std::vector<double> &workspace) const {
  int remaining = 0;
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  printf("Filtering grasps outside of workspace ...\n");

  const candidate::HandGeometry &hand_geometry =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (!is_valid(j)) {
        printf("hand %d of set %d is not valid\n", j, i );
        continue;
      }
      double half_width = 0.5 * hand_geometry.outer_diameter_;
      Eigen::Vector3d left_bottom =
          hands[j]->getPosition() + half_width * hands[j]->getBinormal();
      Eigen::Vector3d right_bottom =
          hands[j]->getPosition() - half_width * hands[j]->getBinormal();
      Eigen::Vector3d left_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d right_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d approach =
          hands[j]->getPosition() - 0.05 * hands[j]->getApproach();
      Eigen::VectorXd x(5), y(5), z(5);
      x << left_bottom(0), right_bottom(0), left_top(0), right_top(0),
          approach(0);
      y << left_bottom(1), right_bottom(1), left_top(1), right_top(1),
          approach(1);
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2),
          approach(2);

      // Ensure the object fits into the hand and avoid grasps outside the
      // workspace.
      if (hands[j]->getGraspWidth() >= min_aperture_ &&
          hands[j]->getGraspWidth() <= max_aperture_ &&
          x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] &&
          y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] &&
          z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) {
        is_valid(j) = true;
        remaining++;
      } else {
        printf("filtering out grasp with xmin:%f, xmax:%f, ymin:%f, ymax:%f, zmin:%f, zmax:%f\n",
        x.minCoeff(), x.maxCoeff(), y.minCoeff(), y.maxCoeff(), z.minCoeff(), z.maxCoeff() );
        is_valid(j) = false;
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates within workspace and gripper width: %d\n",
         remaining);

  return hand_set_list_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::generateGraspCandidates(const util::Cloud &cloud) {
  return candidates_generator_->generateGraspCandidateSets(cloud);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::selectGrasps(
    std::vector<std::unique_ptr<candidate::Hand>> &hands) const {
  printf("Selecting the %d highest scoring grasps ...\n", num_selected_);

  int middle = std::min((int)hands.size(), num_selected_);
  std::partial_sort(hands.begin(), hands.begin() + middle, hands.end(),
                    isScoreGreater);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  for (int i = 0; i < middle; i++) {
    hands_out.push_back(std::move(hands[i]));
    printf(" grasp #%d, score: %3.4f, approach: [%f, %f, %f], position: [%f, %f, %f], full_antipodal: %d, half_antipodal: %d\n", i, hands_out[i]->getScore(),
           hands_out[i]->getApproach()[0], hands_out[i]->getApproach()[1], hands_out[i]->getApproach()[2],
           hands_out[i]->getGraspTop()[0], hands_out[i]->getGraspTop()[1], hands_out[i]->getGraspTop()[2],
           (int)hands_out[i]->isFullAntipodal(), (int)hands_out[i]->isHalfAntipodal());
  }

  return hands_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsDirection(
    std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    const Eigen::Vector3d &direction, const double thresh_rad) {
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  int remaining = 0;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>> &hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        double angle = acos(direction.transpose() * hands[j]->getApproach());
        if (angle > thresh_rad) {
          printf(" filtering out grasp #%d with approach: [%f, %f, %f], angle %f relative to [%f, %f, %f]\n", i, hands[j]->getApproach()[0], hands[j]->getApproach()[1], hands[j]->getApproach()[2],
                 angle, direction.transpose().x(), direction.transpose().y(), direction.transpose().z());
          is_valid(j) = false;
        } else {
          remaining++;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasp candidates with correct approach direction: %d\n",
         remaining);

  return hand_set_list_out;
}

bool GraspDetector::createGraspImages(
    util::Cloud &cloud,
    std::vector<std::unique_ptr<candidate::Hand>> &hands_out,
    std::vector<std::unique_ptr<cv::Mat>> &images_out) {
  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  const candidate::HandGeometry &hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // 2. Filter the candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_approach_direction_) {
    hand_set_list_filtered =
        filterGraspsDirection(hand_set_list_filtered, direction_, thresh_rad_);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hand_set_list_filtered, cloud.getCloudOriginal(),
                              "Filtered Grasps (Approach)", hand_geom);
    }
  }

  // 3. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images_out,
                                 hands_out);

  return true;
}

std::vector<int> GraspDetector::evalGroundTruth(
    const util::Cloud &cloud_gt,
    std::vector<std::unique_ptr<candidate::Hand>> &hands) {
  return candidates_generator_->reevaluateHypotheses(cloud_gt, hands);
}

std::vector<std::unique_ptr<candidate::Hand>>
GraspDetector::pruneGraspCandidates(
    const util::Cloud &cloud,
    const std::vector<std::unique_ptr<candidate::HandSet>> &hand_set_list,
    double min_score) {
  // 1. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list, images, hands);

  // 2. Classify the grasp candidates.
  std::vector<float> scores = classifier_->classifyImages(images);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  // 3. Only keep grasps with a score larger than <min_score>.
  for (int i = 0; i < hands.size(); i++) {
    if (scores[i] > min_score) {
      hands[i]->setScore(scores[i]);
      hands_out.push_back(std::move(hands[i]));
    }
  }

  return hands_out;
}

void GraspDetector::printStdVector(const std::vector<int> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", v[i]);
  }
  printf("\n");
}

void GraspDetector::printStdVector(const std::vector<double> &v,
                                   const std::string &name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%3.2f ", v[i]);
  }
  printf("\n");
}

}  // namespace gpd
