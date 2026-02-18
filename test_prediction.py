from app.utils import predict_fallacy


while True:
    text = input("\nEnter argument (type exit to stop): ").strip()

    if text.lower() == "exit":
        break

    if text == "":
        print("Please enter a valid argument.")
        continue

    label, confidence = predict_fallacy(text)

    print("Predicted:", label)
    print("Confidence:", round(confidence, 4))
