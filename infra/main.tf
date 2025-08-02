
provider "aws" {
  region = var.aws_region
}

resource "aws_iam_role" "lambda_exec" {
  name = "lambda_exec_role"
  assume_role_policy = file("${path.module}/iamsr/trust.json")
}

resource "aws_iam_policy" "lambda_policy" {
  name   = "lambda_policy"
  policy = file("${path.module}/iamsr/policy.json")
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attach" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

resource "aws_lambda_function" "titanic_api" {
  function_name = "titanic-api-lambda"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.lambda_handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 512

  filename         = "../src/lambda.zip"
  source_code_hash = filebase64sha256("../src/lambda.zip")

  environment {
    variables = {
      DYNAMO_TABLE = aws_dynamodb_table.predictions.name
    }
  }
}


resource "aws_dynamodb_table" "predictions" {
  name           = "titanic_predictions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "prediction_id"

  attribute {
    name = "prediction_id"
    type = "S"
  }
}

resource "aws_apigatewayv2_api" "titanic_api" {
  name          = "titanic-api-gw"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.titanic_api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.titanic_api.invoke_arn
  integration_method = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.titanic_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.titanic_api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.titanic_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.titanic_api.execution_arn}/*/*"
}
