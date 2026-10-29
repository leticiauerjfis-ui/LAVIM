LAVIM — Laboratório Virtual de Microscopia
Instruções de instalação e execução

1. O QUE ESTÁ NESTE PACOTE
- lavim_app_corrigido.py: código principal da versão web em Python/Streamlit.
- requirements_lavim.txt: lista de bibliotecas Python necessárias.
- README_LAVIM.txt: estas instruções.

2. REQUISITOS
- Um computador com Python instalado.
- Acesso à internet para instalar as bibliotecas.
- Para usar o assistente de IA, uma chave de API válida da OpenAI e acesso ao serviço.
- O Tesseract OCR também precisa estar instalado e configurado no sistema para que o pytesseract consiga ler texto de imagens.

3. INSTALAÇÃO
Abra o terminal na pasta onde os arquivos foram extraídos e execute:

python -m venv .venv

Ative o ambiente virtual:

Windows:
.venv\\Scripts\\activate

macOS/Linux:
source .venv/bin/activate

Instale as bibliotecas:

python -m pip install --upgrade pip
pip install -r requirements_lavim.txt

4. EXECUTAR O LAVIM
Na pasta do projeto, execute:

streamlit run lavim_app_corrigido.py

O Streamlit deverá mostrar um endereço local para abrir no navegador.

5. CHAVE DA OPENAI
Não compartilhe uma chave de API por e-mail nem a publique no código.
Antes de usar o assistente de IA, configure a chave de forma segura. Uma opção no Streamlit é criar:
.streamlit/secrets.toml

Com o conteúdo:
OPENAI_API_KEY = "SUA_CHAVE_AQUI"

O código precisa estar preparado para ler essa configuração por st.secrets. Se ainda contiver o texto "sua-chave-aqui", o assistente de IA não funcionará até que a configuração seja ajustada.

6. OBSERVAÇÃO SOBRE A CALIBRAÇÃO
A calibração depende de identificar corretamente o texto e o comprimento em pixels da barra de escala na imagem. Confira visualmente o resultado com imagens de escala conhecida antes de usar medidas em um relatório científico. Não use medidas quantitativas se a barra não tiver sido identificada e validada corretamente. Imagens podem ter barras em posições, cores e contrastes diferentes.

7. LIMITAÇÕES
Este pacote contém uma versão Python/Streamlit para navegador. Ele não é um APK Android e não contém, por si só, um projeto nativo completo do Android Studio.
