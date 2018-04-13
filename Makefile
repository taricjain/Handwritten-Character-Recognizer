install:
	javac -g CharacterImage.java
	javac -g CharacterRecognizer.java
	javac -g CharImgType.java
	javac -g MnistData.java
	javac -g Network.java
	java CharacterRecognizer
clean:
	rm *.class
	