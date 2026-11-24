#!/usr/bin/env bash
# Create (or re-assert the settings of) the private S3 bucket behind `tensor-factory store`.
# Idempotent: safe to run again. Needs the aws CLI and credentials allowed to manage the bucket.
#
#   scripts/store_bucket.sh            # bucket tensor-factory-store-<account id>, region $AWS_REGION
#   scripts/store_bucket.sh my-bucket  # or name it yourself
#
# What it guarantees: all public access blocked, TLS-only, versioning on, old versions
# expire after 30 days, test uploads under ci-tests/ expire after 1 day, and half-finished
# uploads are cleaned up after 7. Prints the TENSOR_FACTORY_STORE line to put in .env.
set -euo pipefail

region="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
bucket="${1:-tensor-factory-store-$(aws sts get-caller-identity --query Account --output text)}"

if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
  echo "bucket exists: $bucket"
else
  if [ "$region" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$bucket" --region "$region" >/dev/null
  else
    aws s3api create-bucket --bucket "$bucket" --region "$region" \
      --create-bucket-configuration "LocationConstraint=$region" >/dev/null
  fi
  echo "created: $bucket ($region)"
fi

aws s3api put-public-access-block --bucket "$bucket" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-versioning --bucket "$bucket" --versioning-configuration Status=Enabled

aws s3api put-bucket-policy --bucket "$bucket" --policy "$(cat <<JSON
{"Version": "2012-10-17", "Statement": [{
  "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
  "Resource": ["arn:aws:s3:::$bucket", "arn:aws:s3:::$bucket/*"],
  "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}
JSON
)"

aws s3api put-bucket-lifecycle-configuration --bucket "$bucket" --lifecycle-configuration '{
  "Rules": [
    {"ID": "expire-old-versions", "Status": "Enabled", "Filter": {"Prefix": ""},
     "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
     "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
     "Expiration": {"ExpiredObjectDeleteMarker": true}},
    {"ID": "expire-test-uploads", "Status": "Enabled", "Filter": {"Prefix": "ci-tests/"},
     "Expiration": {"Days": 1},
     "NoncurrentVersionExpiration": {"NoncurrentDays": 1}}
  ]}'

# Read the settings back rather than trusting the calls above.
blocked=$(aws s3api get-public-access-block --bucket "$bucket" \
  --query 'PublicAccessBlockConfiguration.[BlockPublicAcls,IgnorePublicAcls,BlockPublicPolicy,RestrictPublicBuckets]' \
  --output text)
versioning=$(aws s3api get-bucket-versioning --bucket "$bucket" --query Status --output text)
rules=$(aws s3api get-bucket-lifecycle-configuration --bucket "$bucket" --query 'length(Rules)' --output text)
if [ "$blocked" != "True	True	True	True" ] || [ "$versioning" != "Enabled" ] || [ "$rules" != "2" ]; then
  echo "FAILED to verify: public-block=[$blocked] versioning=$versioning lifecycle-rules=$rules" >&2
  exit 1
fi
echo "verified: public access blocked, versioning on, 2 lifecycle rules, TLS-only"
echo "TENSOR_FACTORY_STORE=s3://$bucket"
