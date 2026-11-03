# Reconhecimento de letras de Libras em tempo real com YOLO (fine-tuning)

Pipeline completo: pré-processamento da base → fine-tuning de um detector YOLO11n (pré-treinado no COCO) → avaliação em teste → protótipo com webcam.

## Base de dados
`k4gs3bmx5k-3`: 4.411 imagens 200×200, anotadas em Pascal VOC (1 mão por imagem), 15 letras estáticas:
A B C D E I L M N O R S U V W. Desbalanceada (E: 609 imagens, L/O/U/V/W: 150).

## Estrutura
```
src/prepare_dataset.py   VOC -> YOLO, limpeza, split por blocos, data.yaml, estatísticas
src/train.py             fine-tuning do YOLO11n com data augmentation
src/evaluate.py          métricas no conjunto de teste + acurácia top-1
src/kfold.py             validação cruzada 5-fold agrupada e estratificada
src/make_report.py       gráficos (relatorio/*.png) + relatorio/parametros.txt
src/realtime.py          protótipo com webcam (suavização temporal + soletração)
```

## Como rodar
```bash
pip install -r requirements.txt
python src/prepare_dataset.py --src "C:/Users/Casa/Downloads/k4gs3bmx5k-3" --out datasets/libras_yolo
python src/train.py --epochs 60 --batch 64 --name libras_yolo11n_b64
python src/evaluate.py --weights runs/libras_yolo11n_b64/weights/best.pt
python src/kfold.py --k 5 --epochs 30 --batch 64
python src/make_report.py
python src/realtime.py --weights runs/libras_yolo11n_b64/weights/best.pt
```
Na webcam: `q` sai, `r` liga/desliga o modo ROI (quadrado central), `espaço`/`backspace`/`c` editam o texto.

## Decisões de pré-processamento
- **Limpeza:** 2 XML sem imagem e 1 imagem sem XML descartados; caixas recortadas aos limites da imagem.
- **Conversão:** (xmin, ymin, xmax, ymax) → (classe, xc, yc, w, h) normalizados.
- **Split 70/15/15 estratificado em blocos de 20 frames consecutivos.** As imagens são frames de vídeo
  quase idênticos; um split aleatório colocaria frames vizinhos em treino e teste (vazamento de dados)
  e inflaria as métricas. Resultado: 3.100 treino / 660 validação / 650 teste.

## Treinamento (fine-tuning)
- Modelo: YOLO11n (2,6 M parâmetros), pesos COCO como ponto de partida — a cabeça de detecção é
  reinicializada para 15 classes e toda a rede é ajustada.
- AdamW, lr0 = 0,001 com cosine decay e 3 épocas de warmup, 60 épocas, imgsz 416, early stopping (paciência 15).
- Batch 32 (3,1 GB de GPU) e batch 64, comparados em `relatorio/03_comparacao_batch.png`.
- Augmentation: mosaic (desligado nas 10 últimas épocas), escala ±50%, translação, rotação ±10°,
  espelhamento horizontal e HSV (iluminação). O objetivo é diminuir a diferença entre a base
  (mão ocupando todo o quadro, fundo fixo) e a câmera real.
- Hardware: RTX 3060. Batch 32: 25,7 min (3,1 GB de GPU); batch 64: 15,5 min (4,3 GB).

## Resultados no conjunto de teste (650 imagens nunca vistas)
| Métrica | Batch 32 | Batch 64 (modelo final) |
|---|---|---|
| Precisão | 0,976 | 0,964 |
| Recall | 0,954 | **0,988** |
| mAP@50 | 0,990 | **0,995** |
| mAP@50-95 | 0,906 | **0,917** |
| Acurácia top-1 (letra da detecção mais confiante) | 97,8 % | **99,2 %** |
| Recall da letra U | 0,31 | **0,82** |

Com batch 64, 14 das 15 letras têm recall de 100%. O **U** ainda é confundido com **R** e **V**:
os três sinais têm dois dedos estendidos e diferem só em os dedos estarem juntos, cruzados ou afastados.
No teste, todos os erros vêm de um único trecho de vídeo do U (`relatorio/09_erros_teste.png`).

## Validação cruzada 5-fold (`src/kfold.py`)
Usa treino + validação (3.760 imagens), agrupada em blocos de 20 frames e estratificada por letra,
com 30 épocas por fold e batch 64. Cada modelo é avaliado na validação do próprio fold e no teste fixo.

| Métrica | Validação dos folds (média ± dp) | Teste (média ± dp) |
|---|---|---|
| Precisão | 0,965 ± 0,034 | 0,950 ± 0,025 |
| Recall | 0,969 ± 0,031 | 0,973 ± 0,020 |
| mAP@50 | 0,983 ± 0,013 | 0,981 ± 0,013 |
| mAP@50-95 | 0,892 ± 0,020 | 0,885 ± 0,010 |
| Acurácia top-1 | 0,979 ± 0,023 | 0,977 ± 0,008 |

As confusões ao longo dos folds (`relatorio/08_kfold_matriz_confusao.png`) acontecem entre sinais parecidos:
I↔U/R, M↔N, U→R, V→L e W→V. Algumas letras variam mais entre folds (I, L, R),
porque cada fold tem trechos de vídeo diferentes dessas letras.

## Gráficos e parâmetros (`relatorio/`)
`python src/make_report.py` gera: distribuição da base, curvas de treino (losses, P/R, mAP, LR),
comparação batch 32 x 64, matriz de confusão do teste, métricas por letra, métricas do k-fold por fold,
mAP por letra no k-fold, matriz de confusão somada dos folds, exemplos de erros, distribuição da confiança,
curva precisão-recall e `parametros.txt` com todos os parâmetros (ambiente, pré-processamento, treino, k-fold).

## Limitações
- Base com pouca variedade (um fundo, poucas pessoas, mão centralizada): o desempenho na webcam tende a ser
  menor que no teste. O modo ROI do protótipo aproxima o enquadramento do usado no treino.
- As imagens vêm de uma pasta `asl_alphabet_train`. As 15 letras escolhidas são estáticas e praticamente
  iguais em Libras e ASL. As letras restantes não estão na base: as estáticas F, G, P, Q, T e Y, e as
  com movimento Ç, H, J, K, X e Z, que exigiriam um modelo temporal.
- Para melhorar: coletar imagens próprias com a webcam (outros fundos e pessoas) e fazer um novo fine-tuning a partir do `best.pt`.
