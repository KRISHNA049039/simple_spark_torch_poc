udo su -
export ARTIFACTS_BUCKET=sparkinferenceclusterstack-artifactsbucket2aac5544-zlcd7fgifndh
aws s3 cp s3://$ARTIFACTS_BUCKET/inference/project.zip /opt/spark-inference/project.zip
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app
docker rm -f spark-master spark-cpu-worker 2>/dev/null
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
MASTER_IP=$(hostname -I | awk '{print $1}')
docker run -d --name spark-master --network host multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
sleep 10
docker run -d --name spark-cpu-worker --network host multi-model-inference:latest bash -c "start-worker.sh spark://$MASTER_IP:7077 -c 2 -m 4g && tail -f /opt/spark/logs/*worker*"
On GPU worker:

bash

sudo su -
export ARTIFACTS_BUCKET=sparkinferenceclusterstack-artifactsbucket2aac5544-zlcd7fgifndh
aws s3 cp s3://$ARTIFACTS_BUCKET/inference/project.zip /opt/spark-inference/project.zip
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app
docker rm -f spark-gpu-worker 2>/dev/null
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
MASTER_IP=<MASTER-PRIVATE-IP>
docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g multi-model-inference:latest bash -c "start-worker.sh spark://$MASTER_IP:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
Run benchmark (on master):

bash

docker exec -it spark-master bash -c "SPARK_MASTER_URL=spa
