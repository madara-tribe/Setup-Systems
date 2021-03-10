# !/bin/sh
sudo docker run busybox date
gcloud auth configure-docker
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
REGION=us-central1
export IMAGE_REPO_NAME=sonar_tf_nightly_container
export IMAGE_TAG=sonar_tf
# IMAGE_URI: the complete URI location for Cloud Container Registry
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export JOB_NAME=custom_container_tf_nightly_job_$(date +%Y%m%d_%H%M%S)

# docker push
sudo docker build -f Dockerfile -t $IMAGE_URI ./
sudo docker run $IMAGE_URI --epochs 1
sudo docker push $IMAGE_URI
#gcloud components install beta
gcloud beta ai-platform jobs submit training $JOB_NAME   --region $REGION   --master-image-uri $IMAGE_URI   
--scale-tier BASIC   --   --model-dir=$BUCKET_ID   --epochs=10
#gsutil ls gs://$BUCKET_ID/sonar_*
