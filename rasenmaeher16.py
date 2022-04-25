# hfst
# 25. april 2022
#

import math
import numpy as np

from scipy import integrate
from scipy.optimize import curve_fit
from scipy.stats import erlang

import copy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mlp

import os.path
import pickle

class ListOfPositions():
    def __init__(self, width, height):
        self.listOfPositions=[]
        self.width = width
        self.height = height
    def append(self, position, k):
        self.listOfPositions.append((position, k))
    def get(self):
        return self.listOfPositions
    def len(self):
        return len( self.listOfPositions )
    def getDistribution(self,n):
        n = min( n, len(self.listOfPositions))
        matrix = np.zeros( (self.width, self.height), dtype=int )
        for p, _ in self.listOfPositions[:n]:
            matrix[p] += 1

        distribution = np.zeros( max(matrix.flatten()+1),dtype=int)
        for i in matrix.flatten():
            distribution[i] += 1
        distribution = distribution / (sum(distribution))
        return distribution

    def getDistributions(self, start, stop, step, xrange, yrange ):
        """collects how many squares get's how many visits
            step is not evaluated
        """
#        print("start, stop", start, stop )
        start = max(0, start)
        xmin, xmax = xrange
        ymin, ymax = yrange
        listOfVisits = [self.listOfPositions[i][1][0] for i in range(len(self.listOfPositions))]
        listOfPositions = [self.listOfPositions[i][0] for i in range(len(self.listOfPositions))]
        maxVisits = max( listOfVisits )
#        print( "len(listOfVisits)", len(listOfVisits))
#        print( "maxVisits ", maxVisits )
#        print( "stop ", stop, "len(listOfVisits)", len(listOfVisits[0]) )
        stop = min( stop, len(listOfVisits) )
#        print( "stop=", stop)
#        print( "maxVisits=", maxVisits)
        distributionList = np.zeros( (1, maxVisits+1), dtype=int )
        distributionList[0][0] = self.width * self.height
#        print( "lOP", listOfPositions )
#        for i in range( start, stop ):
#            if( (xmin <= listOfPositions[i][0] < xmax) and
#                (ymin <= listOfPositions[i][1] < ymax ) ):
#                distribution = copy.deepcopy(distributionList[i-0])
#                distribution[ listOfVisits[i]   ] += 1
#                distribution[ listOfVisits[i]-1 ] -= 1
#                distributionList = np.vstack( [distributionList, distribution] )
#        print( distributionList )

        XXdistributionList = np.zeros( (stop-start+1, maxVisits+1) )
        XXdistributionList[0,0] = self.width * self.height
        for i in range(start, stop ):
            if( (xmin <= listOfPositions[i][0] < xmax) and
                (ymin <= listOfPositions[i][1] < ymax ) ):
                XXdistributionList[ i+1 ]= XXdistributionList[i]
                XXdistributionList[ i+1, listOfVisits[i] ] = XXdistributionList[i, listOfVisits[i]] +  1
                XXdistributionList[ i+1, listOfVisits[i]-1 ] = XXdistributionList[i, listOfVisits[i]-1] - 1

#        print("org", distributionList )
#        print("new",  XXdistributionList )
#        print("=================================================")
#        print( distributionList == XXdistributionList )
#        print("=================================================")
        return XXdistributionList


    def getDistributionsX(self, start, stop, step ):
        """collects how many squares get's how many visits
            step is not evaluated
        """
        listOfPositions = np.array( self.listOfPositions, dtype=object )
        listOfVisits = np.transpose( listOfPositions )[1]
        maxVisits = max( listOfVisits )
        stop = min( stop, len(listOfVisits) )
        distributionList = np.zeros( (1, maxVisits+1), dtype=int )
        distributionList[0][0] = self.width * self.height
        for i in range( max(0,start), stop ):
            distribution = copy.deepcopy(distributionList[i-0])
            distribution[ listOfVisits[i]   ] += 1
            distribution[ listOfVisits[i]-1 ] -= 1
            distributionList = np.vstack( [distributionList, distribution] )
        return distributionList
                
    def getDistributionsXX(self,start, stop, step):
        """ returns for each step the relative frequency of squares visited 0, 1, 2, ... time by the robot """
        stop = min( stop, len(self.listOfPositions))
        distributionList= []
        matrix = np.zeros( (self.width, self.height), dtype=int )
        maxlen = len( self.getDistribution( stop ) )
        ialt = 0
        for i in range( start, stop, step ):
#            print( "ialt=", ialt, " i=", i, self.listOfPositions[ialt:i] )
            for p, _ in self.listOfPositions[ialt:i]:
                matrix[p] += 1

            distribution = np.zeros( maxlen , dtype=int)
            for j in matrix.flatten():
                distribution[j] += 1
            distribution = distribution / (sum(distribution))
            distributionList.append( distribution )
            ialt = i
        return distributionList
    
class displays:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        
    def setDistributionList(self, distributionList ):
        self.distributionList = distributionList
    def plotDistribution( distribution ):
        plt.bar( range(len(distribution)), distribution)
        plt.show()

    def animateWayAnimation1(self, i ):
        x1, x2 = np.transpose( self.poslist[i-2:i] )
#        x1, x2 = self.poslist[i-2:i]
        print( "i = ", i,
               "x1=", x1, "x2=", x2,
               self.poslist[i-2:i],
               "type: ", type(x1))
#                    i =  2 [0 1] [1 1] [(0, 1), (1, 1)]
#                    i =  2 [0 1] [1 1] [(0, 1), (1, 1)]
#                    i =  3 [1 2] [1 1] [(1, 1), (2, 1)]
#                    i =  4 [2 3] [1 0] [(2, 1), (3, 0)]
#        line = mlp.lines.Line2D( x, y )
        self.line = plt.plot( x1,x2, color = 'green', marker='o' )
        return line,
    def animateWayAnimation(self, i ):
        p, k = self.poslist[i]
        x, y = p
        self.x.append(x)
        self.y.append(y)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
#        self.text.set_position( [x+0.5,y+0.5] )
        self.text.set_text( "Step " + str(i) + "hits " + str(k))
        return self.line, self.text

    def animate( self, posList, width, height ):
        self.poslist = posList
        fig, ax = plt.subplots()
        ax.set_xlim( 0-1, width )
        ax.set_ylim( 0-1, height )
        self.x=[]
        self.y=[]
        self.line, = ax.plot(0,0,marker='o')
        self.text = ax.text(0, height, "1")
        anim = animation.FuncAnimation(
            fig, self.animateWayAnimation,
            frames = range(len(self.poslist)),
            interval = 1,
            blit=False,
            repeat = False )
        ax.minorticks_on()
        ax.grid(which='both')
        plt.show()
        return anim

    def animateWayInit():
        line = mlp.lines.Line2D( self.poslist[0], self.poslist[0] )
        return line,
        

class Lawn:
    """ Eine Wiese, die gemaeht wird """
    def __init__(self, width, height, start):
        self.pos = start
        self.positions = ListOfPositions(width, height)
        self.arcs=[]
        self.positions.append(self.pos, (1,1) )
        self.width = width
        self.height = height
        self.reset()
        self.setPixel(self.pos)
        self.bresenhamCalls = 0
    def reset(self):
        self.lawn = np.zeros( (self.width, self.height), dtype=int )
#        self.positions = []

    def withinLawn(self, r ):
        column, line = r
    #    print ( column , width , line, height )
        return (column>=0) & (column<self.width) & (line>=0) & (line<self.height)
    def setPixel(self, r0 ):
        column, line = r0
        self.lawn[line, column] += 1
    def getPixel(self, r0):
        column, line = r0
        return self.lawn[line, column]
    
    ## https://de.wikipedia.org/wiki/Bresenham-Algorithmus
    def bresenhamLineWinkel(self, arc ):
        """ Starting from the actual position ran in the direction
            given by arc. Stop when crossing the border.
            As the bresenham-algorithm need a destination a destination
            outside the lawn is calculated.
            For the algorithm see the wikipedia-article
            https://de.wikipedia.org/wiki/Bresenham-Algorithmus            
        """
        self.bresenhamCalls += 1
        r0 = self.pos
        x0, y0 = r0
        direction = np.array( (np.cos(arc), np.sin(arc)) )
        # calculate a destination outside the lawn. The destination
        # must be given as integer.
        r1 = r0 + 2*(self.height+self.width) * direction
        x1, y1 = np.round(r1)
        x1 = int(x1)
        y1 = int(y1)

        dx = abs(x1-x0)
        sx = 1 if x0<x1 else -1
        dy = -abs(y1-y0)
        sy = 1 if y0<y1 else -1
        err = dx + dy

        while( True ):
            e2 = 2 * err;
            if( e2 > dy ):
                err += dy
                x0 += sx
            if( e2 < dx ):
                err += dx
                y0 += sy
            if( self.withinLawn( (x0,y0) ) ):
                    self.setPixel( (x0, y0) )
                    self.positions.append( (x0,y0),
                                           (self.getPixel((x0,y0)),self.bresenhamCalls) )
                    self.pos = (x0, y0)
            else:
#                print( (x0,y0) )
                break
        return self.pos

    def randomWalk(self):
        """ chose random squares
        """
        pos = (np.random.randint( 0, self.width ),
               np.random.randint( 0, self.height ) )
        self.setPixel( pos )
        self.positions.append( pos, self.getPixel(pos) )
                

    def newDirection( self ):
        """ starting from the acutal position calculate a new direction.
            the range is minArc < arc < 180degree - minArc
        """
        spalte, zeile = self.pos
        minArc = 10 * np.pi/180
        res = np.random.rand() # 0 ... 1
        res = (np.pi - (2*minArc)) * res + minArc # 0 ... PI => minArc ... PI - minArc
        if zeile == 0:
            res = res 
        elif zeile == self.height-1:
            res = res + 2 * np.pi / 2 
        elif spalte == 0:
            res = res + 3 * np.pi / 2
        else:
            res = res + 1 * np.pi / 2
        self.arcs.append( (self.pos, res*180/np.pi) )
        return res

    def getPositions(self):
        return self.positions
    def getSize(self):
        return self.height * self.width
        
    def pprint(self):
        print(self.lawn)
        print(self.positions)
    def distribution( self ):
        maxhaeufigkeit = max(wiese.flatten())
        distribution = np.zeros(maxhaeufigkeit+1)
        for i in self.flatten():
            distribution[i] += 1
        distribution = distribution / (sum(distribution))
        return distribution

def doIt():
    global x, y, z
    global ll
    global zy
    global f0, f0aprox
    
### gibt es reproduzierbare Verlaeufe?
### mehrere Verlaeufe mit den gleichen Wiese
    if( True ):
        fig, ax = plt.subplots()
        np.random.seed(1)
        length = 25
        noOfRuns = 25
        f0=[]
        for i in range(noOfRuns):
            x = Lawn( length, length, (0,0) )
            for _ in range(length*length*1):
                x.bresenhamLineWinkel( x.newDirection() )
            y = x.getPositions()
            z = y.getDistributions(0, 5*length*length, 1,
                                   (0, y.width), (0, y.height) )
            f0.append( np.transpose( z )[0] )
            print( "len(np.transpose( z )[0] = ", len(np.transpose( z )[0] ) )
            ax.plot( f0[i] )
        mittelwerte = np.mean( f0, axis=0 )
        abweichungen = np.std( f0, axis=0 )
        
        ax.plot( mittelwerte )
        ax.fill_between(range(len(mittelwerte)),
                mittelwerte+3*abweichungen, mittelwerte-3*abweichungen,
                alpha=0.2)
        ax.set_title("Anzahl der unbesuchten Felder - " + str(noOfRuns) + " Versuche")
        fig.show()
        filename="01a-vergleich"
        fig.savefig(filename+".png")
        fig.savefig(filename+".svg")
        fig.clear()

        fig, ax = plt.subplots()
        ax.plot( mittelwerte, label="Mittelwert" )
        ax.fill_between(range(len(mittelwerte)),
                        mittelwerte+3*abweichungen, mittelwerte-3*abweichungen,
                        alpha=0.2, label="$\pm 3 \sigma$")
        ax.legend()
        ax.set_title("Anzahl der unbesuchten Felder - " + str(noOfRuns) + " Versuche")
        fig.show()
        filename="01b-vergleich"
        fig.savefig(filename+".png")
        fig.savefig(filename+".svg")
        fig.clear()

        fig, ax = plt.subplots()
        ax.plot( abweichungen )
        fig.show()
        filename="02a-abweichungen"
        fig.savefig(filename+".png")
        fig.savefig(filename+".svg")
        fig.clear()

###
### 
###
    np.random.seed(1)


    x=dict()
    y=dict()
    z=dict()
    popt=dict()
    pcov=dict()

    ax1=dict()
    ax2=dict()
    ax3=dict()
    
    fig1, ((ax1[25], ax1[50]), (ax1[70], ax1[100])) = plt.subplots(2,2)
    fig2, ((ax2[25], ax2[50]), (ax2[70], ax2[100])) = plt.subplots(2,2)
    fig3, ((ax3[25], ax3[50]), (ax3[70], ax3[100])) = plt.subplots(2,2)

###
### Verlaeufe mit verschiedenen Dimensionen erzeugen / einlesen
###
    method = "bresenheim"
#    method = "random"
    modelTyp = "Lxe-Funktion"
#    modelTyp = "e-Funktion"

    lengthList = [25, 50, 70, 100]
#    lengthList = [50]

    for length in lengthList:
        x[length] = Lawn(length,length, (0,0) )
        size = length * length
        
        filename=str(length)+"-"+method+".dat"
        if( os.path.exists("y"+filename) and os.path.exists("z"+filename) ):
            try:
                print( "read " + filename )
                file=open("z"+filename, "rb")
                z[length] = np.load(file)
                file.close()
                file=open("y"+filename, "rb")
                y[length] = pickle.load(file)
                file.close()
            except:
                os.remove("z"+filename)
                os.remove("y"+filename)
                raise Exception( "Problems when reading file" )
        else:
            print( "calculate " )
            if (method == "bresenheim"):
#                for _ in range(int(size*size/5)):
                while  (x[length].positions.len() < 5*size):
                    x[length].bresenhamLineWinkel( x[length].newDirection() )
            elif (method == "random"):
                for _ in range( 5*size ):
                    x[length].randomWalk()
            else:
                raise Exception( method + " not implemented" )

            y[length] = x[length].getPositions()
            z[length] = y[length].getDistributions(0, 4*size, 1,
                                                   (0, y[length].width),
                                                   (0, y[length].height) )
            file=open("z"+filename, "wb")
            np.save(file, z[length] )
            file.close
            file=open("y"+filename, "wb")
            pickle.dump( y[length], file )
            file.close
    

### Unnormierte Verlaeufe
    if( True ):
        if (modelTyp == "e-Funktion"):
            def model01( t, w ):
                return np.exp( -w * t ) # w = 1/2500 = 0.0004
            def model02( length ):
                def model02int( t, w ):
                    return np.exp( -w/(length*length) * t ) # w = 1
                return model02int
        elif (modelTyp == "Lxe-Funktion"):
            def model01( t, w ):
                return w * np.exp( -w * t )
            def model02( length ):
                def model02int( t, w ):
                    return w/(length*length) * np.exp( -w/(length*length) * t )
                return model02int
        else:
                raise Exception( modelTyp + " not implemented" )

        for length in lengthList:
            thelabel= method + " " + modelTyp + " : size = " + str(length)
            ax1[length].set_title(" size = " +str(length)+" x "+str(length)+" = "+str(length*length))
            f = np.transpose( z[length] )[0]
            t = range(len(f))
            ax1[length].plot( t, f )
            ax1[length].plot( [ 0, f[0] ], [ f[0],0 ], 'y' )
            ax1[length].grid()
        fig1.suptitle("fig1 : Rohdaten : Methode "+method+" : Model Typ "+ modelTyp )
        fig1.tight_layout()
        fig1.show()

        ax1a=dict()
        fig1a, ((ax1a[25], ax1a[50]), (ax1a[70], ax1a[100])) = plt.subplots(2,2)
        fig,ax = plt.subplots()
        for length in lengthList:
            thelabel= method + " " + modelTyp + " : size = " + str(length)
            ax1a[length].set_title(" size = " +str(length)+" x "+str(length)+" = "+str(length*length))
            f = np.transpose( z[length] )[0] / np.transpose( z[length] )[0,0]
            t = range(len(f)) / np.transpose( z[length] )[0,0]
            ax1a[length].plot( t, f )
            ax1a[length].plot( [ 0, f[0] ], [ f[0],0 ] )
            ax1a[length].grid()
            ax.plot( t, f, label=length )
            ax.plot( [ 0, f[0] ], [ f[0],0 ], 'y' )
            ax.legend()
        fig1a.suptitle("fig1a : Rohdaten : Methode "+method+" : Model Typ "+ modelTyp )
        fig1a.tight_layout()
        fig1a.show()
        ax.grid()
        fig.show()

        
    
        if( True ):
            filename="02a-rohdaten-"+method+modelTyp
            fig1.savefig(filename+".png")
            fig1.savefig(filename+".svg")
        for length in lengthList:
            normierung = np.transpose( z[length] )[0,0]
            print( normierung )
            f0 = np.transpose( z[length] )[0] / normierung
            t = range(len(f0)) / normierung
            print( "f0 : ", f0.size )
            print( "t  : ", t.size )
            popt[length], pcov[length] = curve_fit(model01, t, f0)
            print("fig2 : popt["+str(length)+"]", popt[length], 1.0/size )
            taprox = t
            f0aprox = [model01(tau, popt[length]) for tau in taprox]
            thelabel= method + " " + modelTyp + " : size = " + str(length)
            ax2[length].set_title(" size = " +str(length)+" x "+str(length))
            ax2[length].plot( t, f0, label=method )
            thelabel=  "fit : " + (f"{popt[length][0]:.2e}") + "\nconv : " + (f"{pcov[length][0][0]:.2e}")
            ax2[length].plot( taprox, f0aprox, label=thelabel )
            tline = np.linspace(0, 1/popt[length][0])
            fline = [ 1 - popt[length][0]*tau for tau in tline ]
            ax2[length].plot( tline, fline, label="optimal way")
            ax2[length].legend()
            ax2[length].grid()
            # normierte curve_fit
            popt[length], pcov[length] = curve_fit( model02(length), t, f0,
                                                    p0=(1))
            print("fig3 : popt["+str(length)+"]", popt[length] )
            taprox = t
            f0aprox = [model02(length)(tau, popt[length])[0] for tau in taprox]
            thelabel= method + " : size = " + str(length)
            thelabel=  "fit : " + (f"{popt[length][0]:.2e}") + "\nconv : " + (f"{pcov[length][0][0]:.2e}")
            tline = np.linspace(0, (size)/popt[length][0])
            fline = [ 1 - popt[length][0]*tau/(size) for tau in tline ]
        fig2.suptitle("fig2 : Unnormiert : Methode "+method+" : Model Typ "+ modelTyp )
        fig2.tight_layout()
        fig2.show()
        if( True ):
            filename="02a-unnormiert-"+method+modelTyp
            fig2.savefig(filename+".png")
            fig2.savefig(filename+".svg")


        normierung = np.transpose( z[length] )[0,0]
        f0 = np.transpose( z[length] )[0] / normierung
        t = np.arange( len(f0) ) / normierung
        taprox = t
        f0aprox = [model02(length)(tau, popt[length])[0] for tau in taprox]
        fig, ax = plt.subplots()
        ax.plot( f0 -f0aprox, label="f0-f0aprox" )
        ax.grid()
        ax.set_title( "f0 - f0aprox : " +str(length)+" x "+str(length))
        fig.suptitle("Unnormiert : Methode "+method+" : Model Typ "+ modelTyp )
        fig.show()

        fig, ax = plt.subplots(1)
        ax.plot( np.linspace(20*20,100*100),
                  [1/(x) for x in np.linspace(20*20,100*100)],
                  label="1/x")
        ax.plot( [25*25, 50*50, 70*70, 100*100],
                  [1.63e-3, 3.99e-4, 2.03e-4, 9.93e-5], "o",
                  label="Punkte")
        ax.legend()
#        fig.show()
        if( True ):
            filename="02c-normierung"
            fig.savefig(filename+".png")
            fig.savefig(filename+".svg")
######## Normierung der ordinate
        if (modelTyp == "e-Funktion"):
            def model02( size ):
                def model02int( t, w ):
                    return np.exp( -w/(size*size) * t )
                return model02int

            def model03( n, size ):
                facNm1 = math.factorial(n-1)
                def model1( t, Lambda ):
                    return ((Lambda**n) * (t**(n-1)) / facNm1)*np.exp(-Lambda * t)
                return model1
        elif (modelTyp == "Lxe-Funktion"):
            def model02( size ):
                def model02int( t, w ):
                    return np.exp( -w/(size*size) * t )
                return model02int

            def model03( n, size ):
                facNm1 = math.factorial(n-1)
                def model1( t, Lambda ):
                    return ((Lambda**n) * (t**(n-1)) / facNm1)*np.exp(-Lambda * t)
                return model1


        ax3=dict()
        fig3, ((ax3[25], ax3[50]), (ax3[70], ax3[100])) = plt.subplots(2,2)
        for length in lengthList:
            print( "length = ", length )
            nn = 1 # maximal 12, minimal 1
            for nn in range(1, 4):
                normierung = np.transpose( z[length] )[0,0]
                f0 = np.transpose( z[length] )[nn-1] / normierung
                t = np.arange( len(f0) ) / normierung
                popt, pcov = curve_fit( model03(nn, normierung), t, f0, p0=(1) )
                print("fig3 : popt["+str(nn)+"]", popt )

                print( "normiert", popt )
                ax3[length].plot( t, [model03(nn, 25*25)(tau, *popt) for tau in t],
                                  label= nn)
                ax3[length].plot( t, f0 )
                if( nn == 1):
                    ax3[length].plot( np.linspace(0,1),
                             [1-popt[0]*tau for tau in np.linspace(0,1)],
                             label="optimal")
            ax3[length].legend()
            size = y[length].width * y[length].height
            ax3[length].set_title(" size = " +str(length)+" x "+str(length))
        fig3.suptitle("Erlang : Methode "+method+" : Model Typ "+ modelTyp )
        fig3.tight_layout()
        fig3.show()
        if( True ):
            filename="03-Erlang-" + method+modelTyp
            fig3.savefig(filename+".png")
            fig3.savefig(filename+".svg")


doIt()

###### animationen

lengthList = [25]


for length in lengthList:

    # lOV = [y[25].listOfPositions[i][1][0] for i in range(len(y[25].listOfPositions))]
    size = length*length
    lOP = [y[length].listOfPositions[i][0] for i in range(len(y[25].listOfPositions))]
    print( lOP[:3] )

#    figAnim, (axLawn, axHaeufigkeit) = plt.subplots(
#        1,2, constrained_layout=True )

    figAnim = plt.figure(figsize=[3*4.8, 1*4.8])
    if( False ):
        gs = figAnim.add_gridspec(2, 2)
        axLawn = figAnim.add_subplot( gs[0,0] )
        axHaeufigkeit = figAnim.add_subplot( gs[1,0] )
    else:
        gs = figAnim.add_gridspec(2,6)
        axLawn = figAnim.add_subplot( gs[0:2, 0:2] )
        axHaeufigkeit = figAnim.add_subplot( gs[0:2, 3:6] )
    xLawn, yLawn = [], []
    x2, y2 = [], []

    ErlangAnz = 3
    for _ in range(ErlangAnz+1):
        x2.append([])
        y2.append([])

    xx2, yy2 = [], []

    f0=dict()
    lnH=[]
    for nn in range(0, ErlangAnz):
        print( "len(z[length]) = ", len(z[length]) )
        normierung = np.transpose( z[length] )[0,0]
        f0[nn] = np.transpose( z[length] )[nn] / normierung
        t = np.arange( len(f0[nn]) ) / normierung
        print( "a nn = ", nn )
        print( "a len(t)= ", len(t) )
        print( "normierung=", normierung )
        lnH.append(0)
        lnH[nn], = axHaeufigkeit.plot( [],[], label=str(nn) + " Besuche" )
    lnH.append(0)
    lnH[ErlangAnz], = axHaeufigkeit.plot( [],[],
                                          label="mehr als " + str(ErlangAnz-1) + " Besuche" )
    axHaeufigkeit.legend()
    print( "len(t)=", len(t) )

    ln, = axLawn.plot( [], [], 'gs', markersize=7 )
    
    def init():
        print("init start")
        axLawn.set_xlim(0, length+1)
        axLawn.set_ylim(0, length+1)
        axLawn.set_xticks( np.arange(0,25,1))
        axLawn.set_yticks( np.arange(0,25,1))
        axLawn.set_aspect('equal')
#        axLawn.xaxis.set_visible(False)
#        axLawn.yaxis.set_visible(False)
#        ax.set_frame_on( False )
        axLawn.grid()
        axHaeufigkeit.set_xlim( 0, len(t)/normierung )
        axHaeufigkeit.set_ylim( 0, 1)
        axHaeufigkeit.set_aspect('auto')

        axHaeufigkeit.plot( t, erlang.pdf(t,1), ':', color='tab:blue' ) #linewidth=0.5 )
        axHaeufigkeit.plot( t, erlang.pdf(t,2), ':', color='tab:orange' ) #linewidth=0.5 )
        axHaeufigkeit.plot( t, erlang.pdf(t,3), ':', color='tab:green' ) # linewidth=0.5 )
        axHaeufigkeit.plot( t, 1-( erlang.pdf(t,1)+erlang.pdf(t,2)+erlang.pdf(t,3))
                            , ':', color='tab:red' ) # linewidth=0.5 )
        
        
        print( lnH )
        print("init ende")
        return (ln, *lnH)
    def update(frame):
#        print("update start ", frame )
        xLawn.append( lOP[frame][0]+0.5 )
        yLawn.append( lOP[frame][1]+0.5 )
        ln.set_data( xLawn, yLawn )
        fErlangAnz = 1
        for nn in range(0,ErlangAnz):
            y2[nn].append( f0[nn][frame] )
            x2[nn].append( t[frame] )
            lnH[nn].set_data( x2[nn], y2[nn] )
            fErlangAnz -= f0[nn][frame]
        y2[ErlangAnz].append( fErlangAnz )
        x2[ErlangAnz].append( t[frame] )
##        print( y2[ErlangAnz])
##        print(x2[ErlangAnz]) 
        lnH[ErlangAnz].set_data( x2[ErlangAnz], y2[ErlangAnz] )
##        print( "lnH[0] = ", lnH[0] )
##        print( "lnH[ErlangAnz-1] = ", lnH[ErlangAnz-1] )
##        print( "lnH[ErlangAnz] = ", lnH[ErlangAnz] )
        
##        nn = 0
##        if( np.mod(frame,int(len(t)/40)) == 0 ):
##            yy2.append( f0[nn][frame] )
##            xx2.append( t[frame] )
##        print( "update Ende ")        
        return (ln, *lnH) #lnH[0], lnH[1]

    if( True ):
        print( "vor ani = ", len(lOP)-1)
        frames = len(t)-1
        interval = int(30000/frames)
        print( "interval = ", interval )
        ani = animation.FuncAnimation( figAnim, update, frames=range(0, frames),
                                 init_func=init, interval=interval,
                                       repeat=False, blit=True)
        if( False ):
            figAnim.show()
    #    figAnim.savefig("animation01.mp4")
        else:
            print( "vor figAnim.show()")
            def lumpi( current_frame: int, total_frames: int):
                pass # print(".")
            ani.save('animation01.mp4',progress_callback = lumpi)
            print( "nach figAnim.show()")
