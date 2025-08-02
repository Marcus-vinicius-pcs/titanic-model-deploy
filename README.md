# Titanic Survival Prediction API

API em FastAPI para predição de sobrevivência no Titanic usando modelos de machine learning.

---

## Estrutura do Projeto

```
.
├── src/
│   ├── main.py              # API FastAPI principal
│   ├── utils.py             # Gerenciamento e predição do modelo
│   └── data_processing.py   # Engenharia e pré-processamento de features
│   └── model/               # Modelos e pipelines serializados (.pkl)
├── test/                    # Testes unitários (pytest)
├── notebooks/               # Notebooks de exploração e treinamento
├── infra/                   # Infraestrutura IaC (Terraform)
│   ├── main.tf
│   ├── variables.tf
│   └── inventories/         # Variáveis por ambiente (dev, hom, prod)
├── requirements.txt         # Dependências Python
├── Dockerfile               # Build da imagem Docker da API
└── README.md                # Este arquivo
```

---

## Endpoints da API

- **POST /predict**  
  Recebe dados de um passageiro e retorna predição e probabilidade de sobrevivência.

- **POST /load**  
  Permite upload de um novo modelo `.pkl` para uso imediato pela API.

- **GET /history**  
  Retorna o histórico das últimas predições realizadas.

- **GET /health**  
  Retorna o status da aplicação e do modelo carregado.

---

## Exemplo de Uso

### Predição

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "PassengerId": 1,
       "Pclass": 3,
       "Name": "John Doe",
       "Sex": "male",
       "Age": 22,
       "SibSp": 0,
       "Parch": 0,
       "Ticket": "A/5 21171",
       "Fare": 7.25,
       "Cabin": null,
       "Embarked": "S"
     }'
```

### Upload de Modelo

```bash
curl -X POST "http://localhost:8000/load" \
     -F "file=@novo_modelo.pkl"
```

---

## Execução Local

```bash
pip install -r requirements.txt
cd src
python main.py
```

---

## Execução com Docker

### Build e Run

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

---

## Testes

Execute todos os testes unitários com:

```bash
pytest tests/
```

---

## Infraestrutura como Código (Terraform)

A pasta `infra/` contém scripts Terraform para provisionar:
- Lambda AWS (deploy do código da pasta src)
- API Gateway HTTP (expondo o endpoint /predict)
- DynamoDB (armazenamento de histórico/predições)
- IAM roles e policies

Variáveis de ambiente estão em `infra/inventories/{dev,hom,prod}/terraform.tfvars`.

---

## Características Técnicas

- **Framework:** FastAPI com validação Pydantic
- **Modelos:** Suporte a upload e troca dinâmica de modelos `.pkl`
- **Pré-processamento:** Engenharia de features, imputação, encoding e scaling
- **Testes:** Cobertura unitária com pytest e mocks
- **Infraestrutura:** Pronto para deploy serverless (AWS Lambda + API Gateway)
- **Docker:** Imagem leve, pronta para produção

---

## Observações

- O modelo e pipeline devem ser serializados com pickle e salvos em `src/model/`.
- O endpoint `/predict` espera o mesmo formato de entrada usado no treinamento.
- Para deploy em Lambda, compacte o conteúdo da pasta `src` em um zip (`lambda.zip`).

---