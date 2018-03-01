install:
	javac -g CharacterImage.java
	javac -g CharacterRecognizerExample.java
	javac -g CharImgType.java
	javac -g Network.java
	javac -g MnistData.java
	java CharacterRecognizerExample
clean:
	rm *.class
	