"""
Protótipo em tempo real: reconhece letras de Libras pela webcam com o YOLO treinado.

- Detecta a mão e a letra em cada frame; usa a detecção mais confiante.
- Suavização temporal: votação por maioria nos últimos N frames (evita "piscar" de letra).
- Soletração: se a mesma letra fica estável por ~1 s, ela é adicionada ao texto.
- Modo ROI: a base de treino tem a mão ocupando quase todo o quadro (200x200); o modo ROI
  recorta um quadrado central e manda só ele ao modelo, aproximando a distribuição de treino.

Teclas:  q = sair | r = liga/desliga ROI | espaço = espaço | backspace = apaga | c = limpa texto

Uso:
    python src/realtime.py --weights runs/libras_yolo11n_b64/weights/best.pt --cam 0
"""
import argparse
import time
from collections import Counter, deque

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/libras_yolo11n_b64/weights/best.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=10, help="frames na janela de votação")
    ap.add_argument("--hold", type=float, default=1.0, help="segundos estável para escrever a letra")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"não foi possível abrir a câmera {args.cam}")

    history = deque(maxlen=args.smooth)
    text, roi_mode = "", True
    stable_letter, stable_since, written = None, 0.0, False
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelho: mais natural para quem sinaliza
        H, W = frame.shape[:2]

        if roi_mode:
            side = int(min(H, W) * 0.8)
            ox, oy = (W - side) // 2, (H - side) // 2
            src = frame[oy:oy + side, ox:ox + side]
            cv2.rectangle(frame, (ox, oy), (ox + side, oy + side), (255, 200, 0), 2)
        else:
            ox, oy, src = 0, 0, frame

        res = model.predict(src, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        letter = None
        if len(res.boxes):
            i = int(res.boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[i].tolist())
            letter, conf = model.names[int(res.boxes.cls[i])], float(res.boxes.conf[i])
            cv2.rectangle(frame, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (0, 255, 0), 2)
            cv2.putText(frame, f"{letter} {conf:.2f}", (x1 + ox, max(30, y1 + oy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        history.append(letter)

        # votação por maioria na janela
        voted, votes = Counter(history).most_common(1)[0]
        voted = voted if votes > len(history) // 2 else None

        now = time.time()
        if voted != stable_letter:
            stable_letter, stable_since, written = voted, now, False
        elif voted and not written and now - stable_since >= args.hold:
            text += voted
            written = True

        # HUD
        fps = 1.0 / max(now - t_prev, 1e-6); t_prev = now
        cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Letra: {voted or '-'}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        if voted and not written:
            prog = min(1.0, (now - stable_since) / args.hold)
            cv2.rectangle(frame, (260, 25), (260 + int(150 * prog), 40), (0, 200, 255), -1)
        cv2.putText(frame, f"{fps:4.1f} FPS  ROI:{'on' if roi_mode else 'off'}", (W - 300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, f"Texto: {text}_", (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        cv2.imshow("Libras - YOLO", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            roi_mode = not roi_mode
        elif k == ord("c"):
            text = ""
        elif k == 32:
            text += " "
        elif k == 8:
            text = text[:-1]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
