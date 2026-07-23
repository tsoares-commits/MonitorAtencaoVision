# Missão da Frota: Processamento de Métricas de Atenção Vision

Você é um agente da Frota Claude rodando este repositório. O objetivo desta missão é processar um dataset de vídeos de soldagem para extrair métricas de instabilidade utilizando um encoder ViT treinado.

## Passo a Passo da Execução:

1. **Obtenção dos Dados (Vídeos):**
   Baixe o arquivo ZIP contendo o dataset de vídeos no seguinte link do File Station (QNAP):
   `https://10.1.83.35/share.cgi?ssid=60a8c2a515c74bbc87b357c45dba6a28`
   Você precisará extrair o conteúdo desse ZIP de forma que a pasta resultante seja `Vídeos Metricas Vision/` na raiz do repositório.
   *(Nota: Se o link direto exigir parse de HTML para encontrar o botão de download, sinta-se livre para usar scripts Python ou bash para extrair a URL real do arquivo)*.
2. **Instalação de Dependências:**
   Assegure-se de que os pacotes necessários estão instalados:
   `pip install opencv-python numpy pandas matplotlib seaborn scipy torch torchvision`

3. **Execução do Monitor de Atenção:**
   Para cada vídeo na pasta `Vídeos Metricas Vision/`, rode o script `monitor_atencao_video.py`.
   - Se o nome do vídeo indicar estabilidade (ex: `Bom 8.avi`), use a label `--label estavel`.
   - Se o nome do vídeo indicar instabilidade (ex: `Respingo`, `Gotejamento`, `Aderencia`), use `--label instavel`.
   
   Exemplo de comando:
   `python monitor_atencao_video.py --video "Vídeos Metricas Vision/Bom 8.avi" --label estavel`
   *(Atenção aos espaços nos nomes dos arquivos e aspas).*

4. **Comparação Estatística:**
   Após rodar todos os vídeos, execute o script de comparação:
   `python comparar_classes.py --base_dir out_monitor`

5. **Finalização:**
   Garanta que a pasta `out_monitor` foi populada com todos os CSVs, plots e painéis gerados. O sistema de submissão da frota irá capturar todas as alterações no branch de resultados.
