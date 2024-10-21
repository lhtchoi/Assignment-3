import jetson.inference
import jetson.utils

# Load the object detection network
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the image from file
img = jetson.utils.loadImage("/home/nvidia/jetson-inference/data/images/airplane_fordetection.jpg")

if img is None:  # Check if the image was loaded successfully
	print("Failed to load image.")
else:
	# Perform object detection
	detections = net.Detect(img)

	# Print the detection results
	print(f"Detected {len(detections)} objects in image")
	for detection in detections:
		width = detection.Right - detection.Left
		height = detection.Bottom - detection.Top
		area = width * height
		center_x = (detection.Left + detection.Right) / 2
		center_y = (detection.Top + detection.Bottom) / 2
		print(f"ClassID: {detection.ClassID}, \nConfidence: {detection.Confidence}, \n"
		      f"Left: {detection.Left}, \nTop: {detection.Top}, \n"
		      f"Right: {detection.Right}, \nBottom: {detection.Bottom}, \n" f"Width: {width}, \n" f"Height: {height}, \n" f"Area: {area}, \n" f"Center: ({center_x}, {center_y})\n")

	# Render the image (this will not display anything since it's not a streaming output)
	display = jetson.utils.videoOutput("display://0")  # Use this to visualize if needed
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

	output_image_path = "/home/nvidia/jetson-inference/data/images/assignment_output1.jpg"

	jetson.utils.saveImage(output_image_path, img)  # Save the image with detections
		
	print(f"Output image saved to {output_image_path}")


