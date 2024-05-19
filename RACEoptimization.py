'''
Psuedo code for optimization
alpha=0.602
gamma=0.101
c=0.1
A=10
a=TBD
beta1=0.9
beta2=0.999
epsilon=10e-8
momentum=0
velocity=0
For k=1:n
	ak=a/((k+A)^alpha)
	ck=c/(k^gamma)
	delta=bernoulli_pm1(p, size)
	thetaplus=theta+ck*delta
	thetaminus=theta-ck*delta
	yplus=loss(thetaplus)
	yminus=loss(thetaminus)
	gradient=(yplus-yminus)/(2*ck*delta)
	momentum=beta1*momentum(previous)+(1-beta1)*gradient
	velocity=beta2*velocity(previous)+(1-beta2)*gradient^2
	momentumBiasCorrected=momentum/(1-beta1)
	velocityBiasCorrected=velocity/(1-beta2)
	theta=theta(previous)-(ak*momentumBiasCorrected)/(sqrt(velocityBiasCorrected)+epsilon)  # New position
	theta=min(theta, thetaMax) # Keep within desired bounds
    theta=max(theta, thetaMin)

This algorithim is based off of these papers:
https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
https://arxiv.org/pdf/1412.6980
'''
import numpy as np

def bernoulli(size=1):
    return np.random.choice([1, -1], size=size)

def convertToSimulationCoordinates():
    rowsToSkip = 1
    solidworksCoords = np.loadtxt('solidworksSuspensionPoints.csv', dtype=str, skiprows=rowsToSkip, delimiter=',').astype(float)
    simulationCoords = np.array(solidworksCoords, dtype=object)
    simulationCoords[:, 0] = solidworksCoords[:, 0].astype(int)
    simulationCoords[:, 1] = -solidworksCoords[:, 3]
    simulationCoords[:, 2] = -solidworksCoords[:, 1]
    simulationCoords[:, 3] = solidworksCoords[:, 2]
    string = np.array2string(simulationCoords.astype(str), separator=', ').replace('[', '').replace(']', '').replace("'", '').replace(',\n ', '\n')
    textFile = open('simulationSuspensionPoints.csv', 'w')
    textFile.write(string)
    
def getInitTheta():
	simulationCoords = np.loadtxt('simulationSuspensionPoints.csv', dtype=str, delimiter=',').astype(float)
	return simulationCoords[:, 1:]

def getPrintableTheta(theta):
    simulationCoords = np.loadtxt('simulationSuspensionPoints.csv', dtype=str, delimiter=',').astype(float)
    simulationCoords = np.array(simulationCoords, dtype=object)
    simulationCoords[:, 0] = simulationCoords[:, 0].astype(int)
    simulationCoords[:, 1:] = np.around(theta, decimals=3)
    string = np.array2string(simulationCoords.astype(str), separator=', ').replace('[', '').replace(']', '').replace("'", '').replace(',\n ', '\n')
    return string

def writeTheta(theta, fileName):
    string = getPrintableTheta(theta)
    textFile = open(fileName, 'w')
    textFile.write(string)

def readKPIs(fileLocation):
    kpis = np.loadtxt(fileLocation.replace('"', ''), dtype=str, delimiter=',', skiprows=1)
    kpis = np.array(kpis, dtype=object)
    kpis = kpis[:, [0, 2]]
    kpis[:, 1] = kpis[:, 1].astype(float)
    kpis_dict = dict(zip(kpis[:, 0], kpis[:, 1]))
    return kpis_dict

def loss(suspensionLocation, wheelbase):
    # Calculate values of different KPIs
    print("Paste file location of processed KPIs for new point: ")
    fileLocation = input()
    kpis = readKPIs(fileLocation)
    # Calcualte performance
    runningTotal = 0
    runningTotal += (min(max(0, kpis["Castor trail - with steer"]-35), kpis["Castor trail - with steer"]-20))**2 # Castor trail between 20 and 35mm
    runningTotal += (min(max(0, kpis["Castor angle - with steer"]-10), kpis["Castor angle - with steer"]-5))**2 # Castor angle 5 and 10 degrees
    runningTotal += (min(max(0, kpis["Kingpin inclination - with steer"]-10), kpis["Kingpin inclination - with steer"]-6))**2 # Kingpin angle between 6 and 10 degrees
    scrubRadius = kpis["Scrub radius - with steer"]
    if suspensionLocation == "front":
        runningTotal += (min(max(0, scrubRadius+5), scrubRadius+10))**2 # Scrub radius between -5 and -10mm
    else:
        runningTotal += (min(max(0, scrubRadius-10), scrubRadius-5))**2 # Scrub radius between 5 and 10mm
    runningTotal += (min(max(0, kpis["Wheel centre lateral offset - with steer"]-40), kpis["Wheel centre lateral offset - with steer"]-5))**2 # Wheel center offset between 5 and 40mm
    runningTotal += (min(max(0, kpis["Wheel centre longitudinal offset - with steer"]-10), kpis["Wheel centre longitudinal offset - with steer"]+10))**2     # Wheel center longitudinal offset between -10 and 10mm
    bumpCastor = kpis["Bump castor (knuckle rotation)"]
    if suspensionLocation == "front":
        runningTotal += (min(max(0, bumpCastor-20), bumpCastor))**2 # Bump castor between 0 and 20 degrees
    else:
        runningTotal += (min(max(0, bumpCastor-90), bumpCastor))**2 # Bump castor between 0 and 90 degrees
    wheelCenterRecession = kpis["Kinematic wheel centre recession"]
    if suspensionLocation == "front":
        runningTotal += (min(0, wheelCenterRecession))**2 # Wheel center recession positive
    else:
        runningTotal += (max(0, wheelCenterRecession))**2 # Wheel center recession negative
    bumpSteer = kpis["Bump steer - on centre"]
    if suspensionLocation == "front":
        runningTotal += 4*(min(max(0, bumpSteer), bumpSteer+1))**2 # Bump steer between 0 and -1 degrees
    else:
        runningTotal += 4*(min(max(0, bumpSteer-2), bumpSteer))**2 # Bump steer between 0 and 2 degrees
    runningTotal += (min(max(0, kpis["Bump camber"]-25), kpis["Bump camber"]-10))**2 # Bump camber between 10 and 25 degrees
    runningTotal += (min(max(0, kpis["Contact patch lateral migration"]-150), kpis["Contact patch lateral migration"]-0))**2 # Lateral tire migration below 150mm
    runningTotal += 0.5*(min(0, kpis["Damper ratio"]-0.7))**2 # Damper ratio greater than 0.7
    runningTotal += (min(0, kpis["Lock angle at full right rack travel"]+23))**2 # Lock angle at full right travel greaer than -23 degrees
    rollCenterHeight = kpis["Roll centre height"]
    if suspensionLocation == "front":
        runningTotal += (min(max(0, rollCenterHeight-100), rollCenterHeight))**2 # Roll center height between 0 and 100mm
    else:
        runningTotal += (min(max(0, rollCenterHeight-150), rollCenterHeight-50))**2 # Roll center height between 50 and 150mm
    ackermannPercentage = kpis["Percent ackermann at full rack travel"]
    runningTotal += 2*(min(max(0, ackermannPercentage-90), ackermannPercentage-70))**2 # Ackermann percentage between 70% and 90%
    return runningTotal

def buildArray(data, line, numVariables):
    array = np.zeros((numVariables, 3))
    for count, row in enumerate(data[data.index(line)+1:data.index(line)+numVariables+1]):
        array[count] = np.array(row.strip("[] \n").split()).astype(float)
    return array


try:
    with open("lastSimulationStep.txt") as file:
        if input("Do you want to load the last simulation step? (y/n) ") == "n":
            raise Exception
        data = file.readlines()
        for line in data:
            if not line[0].isalpha():
                continue
            elif "suspensionLocation" in line:
                suspensionLocation = line.split(":")[1].strip()
            elif "numVariables" in line:
                numVariables = int(line.split(":")[1].strip())
            elif "Iteration" in line:
                iterStart = int(line.split(":")[1].strip()) + 1
            elif "wheelbase" in line:
                wheelbase = float(line.split(":")[1].strip())
            elif "momentum" in line:
                momentum = buildArray(data, line, numVariables)
            elif "velocity" in line:
                velocity = buildArray(data, line, numVariables)
            elif "theta" in line:
                theta = buildArray(data, line, numVariables)
            elif "a:" in line:
                a = float(line.split(":")[1].strip())
except:
    print("No previous simulation step found, starting from scratch")
    suspensionLocation = "front" # front or rear
    wheelbase = 3380
    momentum = 0
    velocity = 0
    a = 4 # Gives a step size of ~1mm
    iterStart = 1
    # Theta is the initial position of the suspension points, it is stored in a file to be read in
    convertToSimulationCoordinates()
    theta = getInitTheta()
    numVariables = np.shape(theta)[0]
    iterationLoss = open("iterationLoss.csv", "w")
    iterationLoss.write("Iteration, Loss\n")
    iterationLoss.close()


alpha = 0.602
gamma = 0.101
c = 0.1
A = 10
beta1 = 0.9
beta2 = 0.999
epsilon = 10e-8
for k in range(iterStart, 100):
    ak = a/((k+A)**alpha)
    ck = c/(k**gamma)
    delta = bernoulli(numVariables*3).reshape(-1, 3)
    thetaplus = theta+ck*delta
    writeTheta(thetaplus, "thetaplus.csv")
    print("Input thetaplus into simulation")
    yplus = loss(suspensionLocation, wheelbase)
    thetaminus = theta-ck*delta
    writeTheta(thetaminus, "thetaminus.csv")
    print("Input thetaminus into simulation")
    yminus = loss(suspensionLocation, wheelbase)
    gradient = (yplus-yminus)/(2*ck*delta)
    momentum = beta1*momentum+(1-beta1)*gradient
    velocity = beta2*velocity+(1-beta2)*gradient**2
    momentumBiasCorrected = momentum/(1-beta1**k)
    velocityBiasCorrected = velocity/(1-beta2**k)
    theta = theta-(ak*momentumBiasCorrected)/((velocityBiasCorrected**0.5)+epsilon)
    print("New theta: ", theta)
    print("gradient: ", gradient)
    print("Iteration: ", k, " Loss: ", yplus)
    iterationLoss = open("iterationLoss.csv", "a")
    iterationLoss.write(str(k)+", "+str(yplus)+"\n")
    iterationLoss.close()
    lastSimulationStep = open("lastSimulationStep.txt", "w")
    lastSimulationStep.write("numVariables: " + str(np.shape(theta)[0]) + "\n")
    lastSimulationStep.write("Iteration: " + str(k) + "\n")
    lastSimulationStep.write("suspensionLocation: " + str(suspensionLocation) + "\n")
    lastSimulationStep.write("wheelbase: " + str(wheelbase) + "\n")
    lastSimulationStep.write("a: " + str(a) + "\n")
    lastSimulationStep.write("momentum: \n" + str(momentum) + "\n")
    lastSimulationStep.write("velocity: \n" + str(velocity) + "\n")
    lastSimulationStep.write("theta: \n" + str(theta) + "\n")
    lastSimulationStep.close()

