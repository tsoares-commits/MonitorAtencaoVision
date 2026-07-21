# Missão da Frota: Processamento de Métricas de Atenção Vision

Você é um agente da Frota Claude rodando este repositório. O objetivo desta missão é processar um dataset de vídeos de soldagem para extrair métricas de instabilidade utilizando um encoder ViT treinado.

## Passo a Passo da Execução:

1. **Obtenção dos Dados (Vídeos):**
   Como os vídeos pesados não foram commitados no repositório Git, o usuário configurará uma variável de ambiente ou fornecerá um link do Google Drive no ato da submissão da tarefa com o zip dos vídeos.
   Se os vídeos não estiverem na pasta `Vídeos Metricas Vision/`, verifique se há instruções adicionais no log para baixá-los, ou aborte a missão informando que a pasta de vídeos não foi encontrada.

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
