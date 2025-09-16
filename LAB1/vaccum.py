print("Vacuum Cleaner")

A=[1,1] # Room A: Vacuum present, Dirt present
B=[0,1] # Room B: Vacuum absent, Dirt present

def cleanCheck(A,B):
  if(A[1] == 0 and B[1] == 0):
    return True
  else:
    return False
def clean_room():
  print("Initial State: Room A",A,", Room B",B)

  while not cleanCheck(A,B):

    if(A[0] == 1):
      A[1] = 0
      A[0] = 0
      B[0] = 1
      print("Vacuum cleaner moved to room B. Room A cleaned.")

    elif(B[0] == 1):
      B[1] = 0
      B[0] = 0
      A[0] = 1
      print("Vacuum cleaner moved to room A. Room B cleaned.")

  print("Final State: Room A",A,", Room B",B)

clean_room()
