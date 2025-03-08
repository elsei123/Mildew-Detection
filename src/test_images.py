import cv2
import os

# Define the absolute path of the image folder
folder = os.path.abspath("../cherry-leaves/healthy")

# Check if the folder exists
if not os.path.exists(folder):
    print(f"Error: The folder '{folder}' does not exist.")
else:
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print("No image files found in the folder.")
    else:
        for f in files:
            img_path = os.path.join(folder, f)
            image = cv2.imread(img_path)

            if image is None:
                print(f"❌ Failed to load image: {img_path}")
            else:
                print(f"✅ Successfully loaded: {f}")
                
                # Convert BGR to RGB (optional for correct display)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display the image
                cv2.imshow("Test Image", image)
                cv2.waitKey(0)  # Press any key to close the window
                cv2.destroyAllWindows()
