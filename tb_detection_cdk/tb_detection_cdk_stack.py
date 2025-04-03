from aws_cdk import (
    Stack,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ec2 as ec2,
    RemovalPolicy,
    Duration,
    CfnOutput
)
from constructs import Construct

class TbDetectionStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # 1. Create ECR Repository with proper configuration
        repo = ecr.Repository(
            self, "TbDetectionRepo",
            repository_name="tb-detection-repo",
            image_scan_on_push=True,
            removal_policy=RemovalPolicy.DESTROY,  # Required for auto_delete_images
            auto_delete_images=True,  # Automatically clean up old images
            lifecycle_rules=[{
                "maxImageAge": Duration.days(30)  # Keep images younger than 30 days
            }]
        )

        # 2. Create VPC with NAT gateway for better isolation
        vpc = ec2.Vpc(
            self, "TbDetectionVPC",
            max_azs=3,
            nat_gateways=1,  # Recommended for production
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ]
        )

        # 3. Create ECS Cluster with monitoring
        cluster = ecs.Cluster(
            self, "TbDetectionCluster",
            vpc=vpc,
            container_insights=True  # Enable CloudWatch Container Insights
        )

        # 4. Build and Deploy Service with health checks
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "TbDetectionService",
            cluster=cluster,
            cpu=4096,  # 4 vCPUs
            memory_limit_mib=8192,  # 8GB RAM
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(
                    directory=".",
                    file="Dockerfile",
                    exclude=["cdk.out", "node_modules", ".git"]  # Reduce build context
                ),
                container_port=8000,
                environment={
                    "TF_CPP_MIN_LOG_LEVEL": "2",  # Reduce TensorFlow logs
                    "PYTHONUNBUFFERED": "1"  # Better Python logging
                },
                enable_logging=True
            ),
            public_load_balancer=True,
            health_check_grace_period=Duration.seconds(60),  # Grace period for startup
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True)  # Auto-rollback on failure
        )

        # Configure target group health checks
        service.target_group.configure_health_check(
            path="/health",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3
        )

        # Configure auto-scaling
        scalable_target = service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=4
        )
        scalable_target.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.minutes(3),
            scale_out_cooldown=Duration.minutes(1)
        )

        # Outputs
        CfnOutput(self, "ECRRepositoryURI", value=repo.repository_uri)
        CfnOutput(self, "LoadBalancerURL",
                value=f"http://{service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "ServiceName", value=service.service.service_name)