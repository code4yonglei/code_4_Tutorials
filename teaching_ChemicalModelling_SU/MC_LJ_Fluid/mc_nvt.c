/*
	 Simple source code designed for teaching "Chemical Modelling"

	  Molecular Dynamics simulation of Lennard-Jones Nano-Particles

				Yong-Lei Wang @ Stockholm University

	Building instructions:
	   - cc -O3 -o mc_simu mc_nvt.c
	   - ./mc_simu
	
													v1.0 - 2018-12-01
													v2.0 - 2021-12-21
													
	** Two files: mc_nvt.c and mc_nvt.h
	** Generate one file from simulation: Thermodynamics_MCsim.txt
	** Output 1 PNG figures after running gnuplot script.

*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "mc_nvt.h"

int main(int argc, const char * argv[])
{

	srand((unsigned int)time(NULL));

	Rho = 0.5; // particle density
	Delta = 1.0/1.0; 
	A = 1.0; 
	Cutoff = 2.0; 
	Beta = 1.0;
	Sigma = 1.0;

	FILE *otpt_file;
	otpt_file = fopen("Thermodynamics_MCsim.txt","w");

	clock_t start = clock(); 

	Length = pow(NumberOfParticles/Rho, 1.0/3.0)*Sigma; // simulation box size

	Initialize();
	Cal_Energy_System();

	fprintf(otpt_file, "# Steps E/Np       Pressure   Nacc/Ntempt\n");
	fprintf(otpt_file, "%5d %10.6f %10.6f %10.6f\n", 0,\
		TotalEnergy/NumberOfParticles, Pressure, (double)Naccepted/Nattempts);
	//Performs MC iterations
	for (int i=0; i<NtotIterations; i++){
		MC_iteration();
		Cal_Energy_System();
		fprintf(otpt_file, "%5d %10.6f %10.6f %10.6f\n", i+1,
			TotalEnergy/NumberOfParticles,Pressure, (double)Naccepted/Nattempts);
		if(!((i+1) % NstepsScreenOtPt))
			printf("%5d %10.6f %10.6f %10.6f\n", i+1,\
				TotalEnergy/NumberOfParticles,Pressure, (double)Naccepted/Nattempts);
	}
	putchar('\n');

	printf("Elapsed Time : %.4f \n", ((double)clock()-start)/CLOCKS_PER_SEC);  

	fclose(otpt_file);
	
	return 0;

}


/* *************************** functions **************************** */

double RAND()
{	
	return drand48();
}

void Initialize() {
  	for(int i=0; i<NumberOfParticles; i++)
	{
  		Positions[i].x = RAND()*Length;
  	  	Positions[i].y = RAND()*Length;
  	  	Positions[i].z = RAND()*Length;	
  	}
	printf("%.4f", rand());
}

void Cal_Energy_System_Per_Bead(vector pos, int i, double *En, double *Vir)
{
	double r2, Virij, Enij;
	vector dr;
	int j, k, l, m;
	Enij = Virij = 0.0;
	for(j=0; j<NumberOfParticles; j++)
	{
		dr.x = pos.x-Positions[j].x;
		dr.y = pos.y-Positions[j].y;
		dr.z = pos.z-Positions[j].z;

		for (k=-1; k<2; k++){
			for (l=-1; l<2; l++){
				for (m=-1; m<2; m++){
					if( j==i && k==0 && l==0 && m==0){}
					else{		
						dr.x -= Length*rint(dr.x/Length)+k*Length; // PBC = periodic boundary conditions
						dr.y -= Length*rint(dr.y/Length)+l*Length;
						dr.z -= Length*rint(dr.z/Length)+m*Length;
						r2 = SQR(dr.x)+SQR(dr.y)+SQR(dr.z); 
						if(r2 < SQR(Cutoff*Length)) // cal energy
						{
							Enij += (A*SQR(Sigma)*exp(-pow(r2,0.5)/Sigma))/r2;
							Virij += (A*SQR(Sigma)*exp(-pow(r2,0.5)/Sigma))/r2*((pow(r2,0.5)/Sigma)+2.0);
						}
					}
				}
			}
		}
	}
	*En = Enij;
	*Vir = Virij;
}


// Calculate the total energy in the modelling system
void Cal_Energy_System(void)
{
	double Eni, Viri;
	TotalEnergy = 0.0;
	TotalVirial = 0.0;
	for(int i=0; i<NumberOfParticles; i++)
	{
		Cal_Energy_System_Per_Bead(Positions[i], i, &Eni, &Viri);
		TotalEnergy += Eni;
		TotalVirial += Viri;
	}

	TotalVirial = TotalVirial*0.5; // avoid overcounting
	TotalEnergy = TotalEnergy*0.5;
	Pressure = Rho*CUBE(Sigma)+(1.0/(3.0*CUBE(Length)))*Beta*TotalVirial; 
  
	// tail correction
	Pressure += (2.0/3.0)*SQR(Rho*Sigma)*M_PI*A*exp(-Cutoff*Length/Sigma)*(Cutoff*Length+3.0*Sigma);	
	TotalEnergy += 2.0*M_PI*NumberOfParticles*Rho*A*pow(Sigma,3.0)*exp(-Cutoff*Length/Sigma);
}


void MC_move(int i)
{
	double EnergyNew, VirialNew, EnergyOld, VirialOld;
	vector NewPosition;
	Nattempts += 1.0;

	Cal_Energy_System_Per_Bead(Positions[i],i,&EnergyOld,&VirialOld); // cal old energy

	NewPosition.x = Positions[i].x+(RAND()-0.5)*Delta*Length; // give a random displacement
	NewPosition.y = Positions[i].y+(RAND()-0.5)*Delta*Length;
	NewPosition.z = Positions[i].z+(RAND()-0.5)*Delta*Length;

	Cal_Energy_System_Per_Bead(NewPosition, i, &EnergyNew, &VirialNew); // cal new energy
         
	if(RAND() < exp(-Beta*(EnergyNew-EnergyOld)))
	{
		Naccepted+=1.0;               // movement is accepted
		Positions[i].x=NewPosition.x; // update new position
		Positions[i].y=NewPosition.y;
		Positions[i].z=NewPosition.z;
	}
}


void MC_iteration()
{
	Naccepted=0.0;
	Nattempts=0.0;
	for (int i=0; i<NumberOfParticles; i++)
		MC_move(i);
}

