import matplotlib.pyplot as plt
import time as t 

times= []
mistakes = 0

print("This program will help you to type the word 'hello' faster")
input("Press enter to continue")

while len(times)<5:
    start= t.time()
    word= input("Type the word: ")
    end = t.time()
    time_elapsed = end - start
    
    times.append(time_elapsed)
    if (word.lower() != "hello"):
        mistakes += 1
print("You made "+ str(mistakes)+ " mistakes")
print("Now lets see your Scale: ")
t.sleep(3)

x = [1,2,3,4,5]
y= times
plt.plot(x,y)
legend= ["1","2","3","4","5"]
plt.xticks(x,legend)
plt.xlabel("Attempts")
plt.title("Typing Scale")
plt.ylabel("Time")
plt.show()


