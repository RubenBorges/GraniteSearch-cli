TO BUILD:

1) First, unzip the ".tar.gz" archive file inside of the directory: "ibm-granite30m"

2) To enable automated building and cleaning of the program, enter the following commands from the project root:
                                                                                                                  chmod -x auto.sh
                                                                                                                  chmod -x clean.sh
                                                                                                                  chmod -x build.sh

3) From project root, execute the automated build script by entering the command: ./auto.sh 

4) The program, model and anything else required should already have been installed in the project directory "dist/"
-----------------------------------------

TO USE:

      Usage:   ./dist/GraniteSearch [path_to_gguf_model] "[query]" [file1.txt] [file2.txt] ...
      
      Test Example:                
                    ./dist/GraniteSearch ./dist/ibm-granite30m/ibm-granite \
                        "What does Linda like to do in the backyard?" \
                        dist/docs/ai_programming.txt \
                        dist/docs/baking_bread.txt \
                        dist/docs/cpp_mechanics.txt \
                        dist/docs/linda_the_dog.txt

<img width="1920" height="1037" alt="image" src="https://github.com/user-attachments/assets/0ef3f9b7-cd33-44d5-b4c7-10a7fffcf0be" />


