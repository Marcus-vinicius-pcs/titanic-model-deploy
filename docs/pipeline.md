# Pipeline CI/CD - GitHub Actions

Este documento descreve a esteira de integração contínua e entrega contínua (CI/CD) simulada para a aplicação Titanic API.

## Fluxo da Pipeline

1. **Lint**
   - Executa o flake8 para garantir qualidade e padrão do código Python.
   - Falhas de lint impedem etapas seguintes.

2. **Testes**
   - Executa testes unitários com pytest na pasta `test/`.
   - Garante que alterações não quebrem funcionalidades.

3. **Build**
   - Instala dependências e simula o build da imagem Docker (`docker build -t titanic-api .`).
   - Não faz push real para registro, apenas simula.

4. **Deploy (Simulado)**
   - Simula o deploy do container (`docker run -p 8000:8000 titanic-api`).
   - Não faz deploy real em ambiente externo.

## Como funciona

- A pipeline é disparada em cada push ou pull request para a branch `main`.
- Cada etapa depende do sucesso da anterior.
- O deploy é apenas simulado para fins de documentação e boas práticas.

## Arquivo de workflow

O arquivo está em `.github/workflows/ci.yml` e pode ser ajustado para deploy real em nuvem (AWS, Azure, GCP, etc) conforme necessidade.

---

**Exemplo de comandos usados:**
- Lint: `flake8 src/ --max-line-length=120`
- Testes: `pytest test/`
- Build: `docker build -t titanic-api .`
- Deploy: `docker run -p 8000:8000 titanic-api`

---

Para dúvidas ou ajustes, consulte o arquivo de workflow ou entre em contato com o responsável pelo projeto.
