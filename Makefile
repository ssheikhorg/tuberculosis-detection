up:
	docker build --platform linux/arm64 -t tb-detection .

down:
	docker stop tb-detection
	docker rm tb-detection

run:
	docker run -p 9000:8080 tb-detection

deploy:
	cdk deploy --profile own

#aws ecr get-login-password --region eu-west-2 --profile own | docker login --username AWS --password-stdin 705538025739.dkr.ecr.eu-west-2.amazonaws.com
#docker tag tb-detection:latest 705538025739.dkr.ecr.eu-west-2.amazonaws.com/serverless-tb-detection-fastapi-dev:latest
#docker push 705538025739.dkr.ecr.eu-west-2.amazonaws.com/serverless-tb-detection-fastapi-dev:latest
#cdk bootstrap aws://705538025739/eu-west-2 --profile own