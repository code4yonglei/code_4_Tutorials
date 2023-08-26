#define SQR(x) ((x)*(x))      // Defines the operation x^2
#define CUBE(x) ((x)*(x)*(x)) // Defines the operation x^3

#define NumberOfParticles 100 // number of particle
#define NtotIterations 100    // number of total MC iterations
#define NstepsScreenOtPt 10   // number of steps to output info on screen (or external file if using >)

typedef struct {
  double x;
  double y;
  double z;
} vector;

/* ************************** VARIABLES ***************************** */
vector Positions[NumberOfParticles];
double Sigma, A, Cutoff;
double Beta;
double Rho, Length;
double Eni, Viri;
double TotalEnergy, EnergyPerParticle;
double TotalVirial, Pressure;
double Etail, Ptail;
double AverageEnPerPar, AveragePressure;
double Delta;
int Naccepted, Nattempts;
double Acceptance;


/* ************************** FUNCTIONS ***************************** */
double RAND();  //Generates a Random Number between [0,1]
void Initialize();  //Initializes simulation, particles start at random positions 
void Cal_Energy_System(); // cal total system energy                                       
void Cal_Energy_System_Per_Bead(vector Positions, int, double *En, double *Vir); // cal energy per particle
void MC_iteration(); // Attemps to make N MC movements (N=NumberOfParticles)
void MC_move(int); // Attemps to displace the k-th particle.
