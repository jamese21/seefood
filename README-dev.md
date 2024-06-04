#To run notebook:
1. conda activate tf
2. cd in seefood directory
3. jupyter notebook
^ Both in conda powershell (search anaconda powershell prompt in search)
4. make sure kernel is jupyter server one at url given when server is run in step 2

#To run docker container:
```
docker run --hostname=31c062ae1f09 --mac-address=02:42:ac:11:00:02 --env=MODEL_NAME=seefood-model --env=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=MODEL_BASE_PATH=/models --volume=C:\Users\User\Documents\Projects\ML-Practice\trying-things\seefood/model:/models/seefood-model --network=bridge -p 8501:8501 --restart=no --label='maintainer=gvasudevan@google.com' --label='org.opencontainers.image.ref.name=ubuntu' --label='org.opencontainers.image.version=20.04' --label='tensorflow_serving_github_branchtag=2.16.1' --label='tensorflow_serving_github_commit=0e6261315e2a8c529842929f5ceeb66b63264e7b' --runtime=runc -t -d tensorflow/serving
```

#To send image to model for predictions (once running in container):
Run send-request.py and modify path to desired image

Completed:
Cleaned data, trained/tested model
Packaged model using tensorflow serving/docker
Successfully hosted docker container at localhost
Got predictions from hosted model!



