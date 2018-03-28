install:
	javac -g CharacterImage.java
	javac -g CharacterRecognizerExample.java
	javac -g CharImgType.java
	javac -g MnistData.java
	javac -g Network.java
	java CharacterRecognizerExample
clean:
	rm *.class
	