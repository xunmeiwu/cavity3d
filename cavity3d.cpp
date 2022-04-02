#include "palabos3D.h"
#include "palabos3D.hh"
#include <vector>
#include <cmath>
#include <iostream>
//#include <sstream>
//#include <fstream>
#include <string>
#include <iomanip>

using namespace plb;
using namespace plb::descriptors;
using namespace std;

typedef double T;
#define DESCRIPTOR descriptors::MRTD3Q19Descriptor

/** 案例参数 **/
const string caseNameChar = "Cavity"; //案例名称，log文件及vtk文件保存名.
const T X_Length = 0.15;            //流域X方向长度；
const T Y_Length = 0.15;            //流域Y方向长度；
const T Z_Length = 0.15;            //流域Z方向长度；

/** 单位转换参数 **/
const T charL = 0.15;                //代表物理长度
const T charU = 0.08;               //代表物理速度；
const T N = 127.;                   //代表物理长度的网格数
const T Re = 12000.;                   //4635.546; //雷诺数
const T latticeU = 0.0866;        //晶格速度，与马赫数匹配

/** 模拟参数 **/
const T iniT = (T)0;                 // 模拟初始时刻，单位s，一般从0开始，若大于0则需要读取现有晶格
const T maxT = (T)iniT + 1466. + 0.1; // 模拟截止时刻，单位s
const T cSmago = 0.12;              // Smagorisky常数

const T statT = (T)0.1;     // 控制台Log显示时间间隔，单位s
const T vtkStartT = (T)0; //VTK文件输出开始时间，单位s
const T vtkT = (T)1466;    // VTK文件输出时间间隔，单位s
const T imSave = (T)733;    // image文件输出时间间隔，单位s
const bool useAve = true;    //是否进行时间平均
const T aveStartT = (T)937; //平均开始时间
const T aveDelta = (T)0.01; //平均时间间隔，单位s

const plint bcType = 1;    //边界条件：1:interp，2:local。默认interp

const bool latSave = true;                                                  //是否保存lattice文件
const T latSaveT = (T)936;                                                  // lattice保存时间间隔，单位s
const bool latLoad = false;                                                  //是否读取lattice文件
const T latLoadT = iniT;                                                    // lattice读取开始时间，单位s
std::string latSaveFile = "checkpointing_", latLoadFile = "checkpointing_"; //写入与读取lattice文件的文件名

const T rho0 = 1.0;
const Array<T, 3> u0((T)0., (T)0., (T)0.);  //默认u0数值

static T SMALL = 1.0e-30;//doubleScalar.H
T aveFrequncy = SMALL; //This keeps the sampling amount for the time average

/** Get formated current system time and **/
string getTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return tmp;
}

/** 确定模型每个边的边界条件 **/
void boundarySetup(MultiBlockLattice3D<T, DESCRIPTOR> &lattice,
                   IncomprFlowParam<T> const &parameters,
                   OnLatticeBoundaryCondition3D<T, DESCRIPTOR> &boundaryCondition)
{
    const plint nx = parameters.getNx();
    const plint ny = parameters.getNy();
    const plint nz = parameters.getNz();

    Box3D Roof(0, nx - 1, 0, ny - 1, nz - 1, nz - 1);		//Roof
    Box3D Ground(0, nx - 1, 0, ny - 1, 0, 0);				//Ground
    Box3D Leftside(0, 0, 0, ny - 1, 1, nz - 2);				//Leftside
    Box3D Rightside(nx - 1, nx - 1, 0, ny - 1, 1, nz - 2); 	//Rightside
    Box3D Frontside(1, nx - 2, 0, 0, 1, nz - 2);			//Frontside
    Box3D Backside(1, nx - 2, ny - 1, ny - 1, 1, nz - 2);	//Backside

    //入口dirichlet速度条件
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Roof, boundary::dirichlet);
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Ground, boundary::dirichlet);
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Leftside, boundary::dirichlet);
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Rightside, boundary::dirichlet);
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Frontside, boundary::dirichlet);
    boundaryCondition.setVelocityConditionOnBlockBoundaries(lattice, Backside, boundary::dirichlet);

//    defineDynamics(lattice, Ground, new plb::BounceBack<T, DESCRIPTOR>);
//    defineDynamics(lattice, Leftside, new plb::BounceBack<T, DESCRIPTOR>);
//    defineDynamics(lattice, Rightside, new plb::BounceBack<T, DESCRIPTOR>);
//    defineDynamics(lattice, Frontside, new plb::BounceBack<T, DESCRIPTOR>);
//    defineDynamics(lattice, Backside, new plb::BounceBack<T, DESCRIPTOR>);

    //所有位置给定初始速度和密度并初始化
    initializeAtEquilibrium(lattice, lattice.getBoundingBox(), rho0, u0);
    setBoundaryVelocity(lattice, Roof, Array<T, 3>(latticeU, (T)0., (T)0.));
    lattice.initialize();
}

/** Calculate the dynamic viscosity of smagorinsky model, nut**/
template <typename T, template <typename U> class Descriptor>
class calcSmagoViscosity : public BoxProcessingFunctional3D_LS<T, Descriptor, T>
{
public:
    calcSmagoViscosity(T nu0_, T cSmago_)
            : nu0(nu0_), cSmago(cSmago_)
    {
    }
    virtual void process(Box3D domain, BlockLattice3D<T, Descriptor> &lattice,
                         ScalarField3D<T> &EffectiveViscosity)
    {
        Dot3D offset = computeRelativeDisplacement(EffectiveViscosity, lattice);
        for (plint iX = domain.x0; iX <= domain.x1; ++iX)
        {
            for (plint iY = domain.y0; iY <= domain.y1; ++iY)
            {
                for (plint iZ = domain.z0; iZ <= domain.z1; ++iZ)
                {
                    Cell<T, Descriptor> &cell = lattice.get(iX + offset.x, iY + offset.y, iZ + offset.z);

                    T rhoBar;
                    Array<T, Descriptor<T>::d> j;
                    Array<T, SymmetricTensor<T, Descriptor>::n> PiNeq;
                    cell.getDynamics().computeRhoBarJPiNeq(cell, rhoBar, j, PiNeq);
                    T preFactor = SmagoOperations<T, Descriptor>::computePrefactor(cell.getDynamics().getOmega(), cSmago);
                    T omegaTotal = SmagoOperations<T, Descriptor>::computeOmega(
                            cell.getDynamics().getOmega(), preFactor, rhoBar, PiNeq);
                    T nuTotalFromSmago = std::max(((T)1. / omegaTotal - 0.5) * DESCRIPTOR<T>::cs2, nu0);
                    EffectiveViscosity.get(iX, iY, iZ) = nuTotalFromSmago-nu0;
                }
            }
        }
    }
    virtual calcSmagoViscosity<T, Descriptor> *clone() const
    {
        return new calcSmagoViscosity(*this);
    }
    void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
        modified[0] = modif::allVariables;
        modified[1] = modif::staticVariables;
    }
    virtual BlockDomain::DomainT appliesTo() const
    {
        return BlockDomain::bulk;
    }

private:
    T nu0;
    T cSmago;
};

/** VTK file output **/
void writeVTK(MultiBlockLattice3D<T, DESCRIPTOR> &lattice,
              IncomprFlowParam<T> const &parameters, plint iter,
              MultiTensorField3D<T, 3> &velocity,
              MultiScalarField3D<T> &rho,
              MultiScalarField3D<T> &nut,
              MultiTensorField3D<T, 3> &uSum,
              MultiTensorField3D<T, 6> &uPrimeSum,
              MultiScalarField3D<T> &nutSum,
              MultiScalarField3D<T> &nutSquareSum,
              MultiScalarField3D<T> &rhoSum,
              MultiScalarField3D<T> &rhoSquareSum,
              MultiTensorField3D<T,3> &uSquareSum
)
{
    T dx = parameters.getDeltaX();
    T dt = parameters.getDeltaT();
    T pressureScale = (dx * dx) / (dt * dt) * DESCRIPTOR<T>::cs2;

    //输出各种瞬时值到vtk文件
    //Obtain the iteration time
    std::ostringstream strOS;
    strOS << fixed << setprecision(6) << iter * parameters.getDeltaT();
    std::string strIterTime = strOS.str();
    VtkImageOutput3D<T> vtkOut("vtk" + strIterTime + "s", dx);
    vtkOut.writeData<3, T>(velocity, "velocity", dx / dt);
    vtkOut.writeData<T>(rho, "rho", 1.);
    vtkOut.writeData<T>(rho, "pressure", pressureScale);
    vtkOut.writeData<T>(nut, "nut", util::sqr(dx) / dt);

    //Calculate meanVelocity
    std::unique_ptr<MultiTensorField3D<T, 3>> uMean = multiply((T)1. / aveFrequncy, uSum, lattice.getBoundingBox());

    //Calculate uPrimeInstantaneous <u'u'><v'v'><w'w'><u'v'><u'w'><v'w'>
    std::unique_ptr<MultiTensorField3D<T, 6>> uPrimeInstantaneous = computeInstantaneousReynoldsStress(velocity, *uMean, lattice.getBoundingBox());
    vtkOut.writeData<6, T>(*uPrimeInstantaneous, "uPrimeInstantaneous", util::sqr(dx / dt));

    //**************************************************************************************

    if (useAve && iter >= parameters.nStep(aveStartT))
    {
        //Calculate uPrime2Mean <u'u'><v'v'><w'w'><u'v'><u'w'><v'w'>
        std::unique_ptr<MultiTensorField3D<T, 6>> uPrime2Mean = multiply((T)1. / aveFrequncy, uPrimeSum, lattice.getBoundingBox());

        //Calculate meanNut <nut>
        std::unique_ptr<MultiScalarField3D<T>> nutMean = multiply((T)1. / aveFrequncy, nutSum, lattice.getBoundingBox());

        //Calculate nutPrime2Mean, <Nut'^2>=<Nut^2>-<Nut>^2
        std::unique_ptr<MultiScalarField3D<T>> nutPrime2Mean = subtract(*multiply((T)1. / aveFrequncy, nutSquareSum, lattice.getBoundingBox()),
                                                                        *multiply(*nutMean, *nutMean, lattice.getBoundingBox()),
                                                                        lattice.getBoundingBox());

        //Calculate rhoMean <rho>
        std::unique_ptr<MultiScalarField3D<T>> rhoMean = multiply((T)1. / aveFrequncy, rhoSum, lattice.getBoundingBox());

        //Calculate rhoPrime2Mean <Rho'^2>=<Rho^2>-<Rho>^2
        std::unique_ptr<MultiScalarField3D<T>> rhoPrime2Mean = subtract(*multiply((T)1. / aveFrequncy, rhoSquareSum, lattice.getBoundingBox()),

                                                                        *multiply(*rhoMean, *rhoMean, lattice.getBoundingBox()),
                                                                        lattice.getBoundingBox());
        //Calculate ((X1^2+X2^2+X3^2+...+Xn^2)/n)^(1/2)
        std::unique_ptr<MultiTensorField3D<T,3>> RMS = computeSqrt (*multiply((T)1. / aveFrequncy, uSquareSum, lattice
                .getBoundingBox()), lattice.getBoundingBox());

        //VTK output
        vtkOut.writeData<3, T>(*uMean, "uMean", dx / dt);
        vtkOut.writeData<6, T>(*uPrime2Mean, "uPrime2Mean", util::sqr(dx / dt));
        vtkOut.writeData<T>(*rhoMean, "pMean", pressureScale);
        vtkOut.writeData<T>(*rhoPrime2Mean, "pPrime2Mean", pressureScale * util::sqr(dx / dt));
        vtkOut.writeData<T>(*nutMean, "nutMean", util::sqr(dx) / dt);
        vtkOut.writeData<T>(*nutPrime2Mean, "nutPrime2Mean", util::sqr(util::sqr(dx) / dt));
        vtkOut.writeData<T>(*rhoMean, "rhoMean", 1.);
        vtkOut.writeData<T>(*rhoPrime2Mean, "rhoPrime2Mean", 1.);
        vtkOut.writeData<3, T>(*RMS,"Urms",dx / dt);

    }
}


/** 计算主程序 **/
int main(int argc, char *argv[])
{
    plbInit(&argc, &argv);
    global::directories().setOutputDir("./tmp/");

    //确定转换参数
    IncomprFlowParam<T> parameters(
            (T)latticeU,      // latticeU
            (T)Re,            // Re
            N,                // N,Resolution
            X_Length / charL, // lx
            Y_Length / charL, // ly
            Z_Length / charL  // lz
    );

    const plint nx = parameters.getNx();
    const plint ny = parameters.getNy();
    const plint nz = parameters.getNz();
    const T dx = parameters.getDeltaX();
    const T dt = parameters.getDeltaT();

    writeLogFile(parameters, caseNameChar); //输出转换参数
    pcout << "omega= " << parameters.getOmega() << std::endl;
    pcout << "Tao= " << parameters.getTau() << std::endl;
    pcout << "LatticeNu= " << parameters.getLatticeNu() << std::endl;
    pcout << "latticeU= " << parameters.getLatticeU() << std::endl;
    pcout << "PhysicalU= " << parameters.getPhysicalU() << std::endl;
    pcout << "getRe= " << parameters.getRe() << std::endl;
    pcout << "PhysicalLength= " << parameters.getPhysicalLength() << std::endl;
    pcout << "Resolution= " << parameters.getResolution() << std::endl;
    pcout << "DeltaX= " << parameters.getDeltaX() << std::endl;
    pcout << "DeltaT= " << parameters.getDeltaT() << std::endl;
    pcout << "Nx=" << nx << ", Ny=" << ny << ", Nz=" << nz << std::endl;

    //定义晶格lattice
    MultiBlockLattice3D<T, DESCRIPTOR> *lattice = new MultiBlockLattice3D<T, DESCRIPTOR>(parameters.getNx(), parameters.getNy(), parameters.getNz(),
            ///new CompositeDynamics<T, DESCRIPTOR>(new IncMRTdynamics<T,DESCRIPTOR>(parameters.getOmega()));
            ///new IncBGKdynamics<T,DESCRIPTOR>(parameters.getOmega()) );
            ///new SmagorinskyRegularizedDynamics<T,DESCRIPTOR>(parameters.getOmega(), cSmago));
         new SmagorinskyMRTdynamics<T,DESCRIPTOR>(parameters.getOmega(), cSmago));
    /// new SmagorinskyMRTdynamics<T, DESCRIPTOR>(parameters.getOmega(), cSmago));
    ///new MRTdynamics<T, DESCRIPTOR>(parameters.getOmega()));
    ///new ConstRhoBGKdynamics<T,DESCRIPTOR>(parameters.getOmega()) );

    //define basic instantaneous physical quantities
    MultiTensorField3D<T, 3> velocity(nx, ny, nz, u0); //velocity
    MultiScalarField3D<T> nut(nx, ny, nz, (T)0.);                 //nut
    MultiScalarField3D<T> rho(nx, ny, nz, (T)0.);                 //rho


    //Define all tensor and scalar fields for averaged operation
    MultiTensorField3D<T, 3> uSum(nx, ny, nz, u0);
    MultiTensorField3D<T, 6> uPrimeSum(nx, ny, nz);
    MultiScalarField3D<T> nutSum(nx, ny, nz, (T)0.);
    MultiScalarField3D<T> nutSquareSum(nx, ny, nz, (T)0.);
    MultiScalarField3D<T> rhoSum(nx, ny, nz, (T)0.);
    MultiScalarField3D<T> rhoSquareSum(nx, ny, nz, (T)0.);

    MultiTensorField3D<T,3> uSquareSum(nx,ny,nz,u0);



    //定义边界条件
    OnLatticeBoundaryCondition3D<T, DESCRIPTOR> *boundaryCondition;
    pcout << "Setting BoundaryCondition:";
    switch (bcType)
    {
        case 1:
            boundaryCondition = createInterpBoundaryCondition3D<T, DESCRIPTOR>();
            pcout << "InterpolationBC3D...OK" << endl;
            break;
//        case 2:
//            boundaryCondition = createLocalBoundaryCondition3D<T, DESCRIPTOR>();
//            pcout << "LocalBC3D...OK" << endl;
//            break;
        default:
            boundaryCondition = createInterpBoundaryCondition3D<T, DESCRIPTOR>();
            pcout << "InterpolationBC3D...OK" << endl;
    }

    boundarySetup(*lattice, parameters, *boundaryCondition);
    delete boundaryCondition;

    std::ostringstream strOS;

    // 读取之前的晶格数据
    if (latLoad)
    {
//        loadRawMultiBlock(lattice, "checkpoint.dat");
        strOS << latLoadT;
        std::string strLoadTime = strOS.str();
        pcout << "Loading lattice file " << latLoadFile + strLoadTime + "s.dat"
              << " ...";
        loadBinaryBlock(*lattice, latLoadFile + strLoadTime + "s.dat");
        pcout << "OK." << endl;
    }

    global::timer("mainLoop").start(); //主循环计时器
    global::timer("iteLog").start();   // log显示之间的循环计时器

    // 从初始时间iniT开始进行主循环
    pcout << "Starting iteration, Current time: " << getTime() << endl;
    for (plint iT = iniT / parameters.getDeltaT(); iT * parameters.getDeltaT() < maxT; ++iT)
    {

        //Update new basic physical quantities
        if (iT % parameters.nStep(statT) == 0 ||
            (useAve && iT % parameters.nStep(aveDelta) == 0 && iT >= parameters.nStep(aveStartT) && iT > parameters.nStep(iniT)) ||
            (iT % parameters.nStep(vtkT) == 0 && iT >= parameters.nStep(vtkStartT) && iT > parameters.nStep(iniT)) ||
            iT % parameters.nStep(imSave) == 0)
        {
            rho = *computeDensity(*lattice, lattice->getBoundingBox());
            velocity = *computeVelocity(*lattice, lattice->getBoundingBox());
            applyProcessingFunctional(new calcSmagoViscosity<T, DESCRIPTOR>(parameters.getLatticeNu(), cSmago), lattice->getBoundingBox(), *lattice, nut);
        }

        /// 输出控制台信息
        if (iT % parameters.nStep(statT) == 0)
        {
            pcout << endl;
            pcout << "step " << iT << "; t=" << iT * parameters.getDeltaT() << endl;
            Array<T, 3> nearGroundVel;
            Array<T, 3> nearRoofVel;
            for (plint i = 0; i < 3; ++i)
            {
                nearGroundVel[i] = (*extractComponent(velocity, i)).get(nx / 2, ny / 2, 1);
                nearRoofVel[i] = (*extractComponent(velocity, i)).get(nx / 2, ny / 2, nz-2);
            }

            pcout << "nearGroundVel= " << fixed << setprecision(6) << nearGroundVel[0]*dx/dt << "," << nearGroundVel[1]*dx/dt << "," << nearGroundVel[2]*dx/dt << ";";
            pcout << "nearRoofVel= " << fixed << setprecision(6) << nearRoofVel[0]*dx/dt << "," << nearRoofVel[1]*dx/dt << "," << nearRoofVel[2]*dx/dt << std::endl;
        }

        /// 执行前进与碰撞函数
        lattice->collideAndStream();

        ///输出image
        if (iT % parameters.nStep(imSave) == 0)
        {
            pcout << "Writing Gif ...";
            const plint imSize = 600;
            Box3D slice(0, nx-1, ny/2, ny/2, 0, nz-1);
            ImageWriter<T> imageWriter("leeloo");
            imageWriter.writeScaledGif( createFileName("ux", iT, 6),
                                        *computeVelocityComponent(*lattice, slice, 0),
                                        imSize, imSize );
            imageWriter.writeScaledGif( createFileName("uz", iT, 6),
                                        *computeVelocityComponent(*lattice, slice, 2),
                                        imSize, imSize );
            imageWriter.writeScaledGif( createFileName("velNorm", iT, 6),
                                        *computeVelocityNorm (*lattice, slice),
                                        imSize, imSize );
            pcout << "OK." << endl;
        }

        ///输出VTK文件
        if (iT % parameters.nStep(vtkT) == 0 && iT >= parameters.nStep(vtkStartT) && iT > parameters.nStep(iniT))
        {
            pcout << "Saving VTK file ...";
            writeVTK(*lattice, parameters, iT, velocity,
                     rho, nut, uSum, uPrimeSum, nutSum,
                     nutSquareSum, rhoSum, rhoSquareSum,
                     uSquareSum);
            pcout << "OK." << endl;
        }

        ///计算平均值
        if (useAve && iT % parameters.nStep(std::max(parameters.getDeltaT(), aveDelta)) == 0 && iT >= parameters.nStep(aveStartT) && iT > parameters.nStep(iniT)) //iT>parameters.nStep(iniT)是必须的，防止初始读入数据0参与平均
        {
            if (iT % parameters.nStep(statT) == 0)
            {
                pcout << "Calculate Average Velocity ...";
            }
            aveFrequncy = aveFrequncy + 1;
            //Sum instantaneous velocity into uSum
            add(uSum, velocity, uSum, lattice->getBoundingBox());
            //Calculate uMean
            std::unique_ptr<MultiTensorField3D<T, 3>> uMean = multiply((T)1. / aveFrequncy, uSum, lattice->getBoundingBox());
            //Calculate instantaneous UPrime u'u', v'v', w'w', u'v', u'w', v'w'
            std::unique_ptr<MultiTensorField3D<T, 6>> currentUPrime = computeInstantaneousReynoldsStress(velocity, *uMean, lattice->getBoundingBox());
            //Sum instantaneous UPrime into uPrimeSum
            addInPlace(uPrimeSum, *currentUPrime);
            //Sum instantaneous nut into nutSum
            addInPlace(nutSum, nut);
            //Sum nutSquare into nutSquareSum
            addInPlace(nutSquareSum, *multiply(nut, nut, lattice->getBoundingBox()));
            //Sum rho into rhoSum
            addInPlace(rhoSum, rho);
            //Sum rhoSquare into rhoSquareSum
            addInPlace(rhoSquareSum, *multiply(rho, rho, lattice->getBoundingBox()));
            //Sum velocitySquare into velocitySquareSum
            addInPlace(uSquareSum,*multiply(velocity, velocity, lattice->getBoundingBox()));


            if (iT % parameters.nStep(statT) == 0)
            {
                pcout << "OK" << endl;
            }
        }

        ///保存晶格状态
        if (latSave && iT % parameters.nStep(latSaveT) == 0 && iT > parameters.nStep(iniT))
        {
//            saveRawMultiBlock(lattice, "checkpoint.dat");
            std::ostringstream strOS1;
            strOS1 << fixed << setprecision(6) << iT * parameters.getDeltaT();
            //strIterTime = strOS1.str();
            std::string strLatticeSave = latSaveFile + strOS1.str() + "s.dat";
            pcout << "Saving the lattice file: " << strLatticeSave << " ...";
            saveBinaryBlock(*lattice, strLatticeSave);
            pcout << "OK." << endl;
        }

        if (iT % parameters.nStep(statT) == 0)
        {
            pcout << "av energy="
                  << setprecision(10) << getStoredAverageEnergy<T>(*lattice)
                  << "; av rho="
                  << setprecision(10) << getStoredAverageDensity<T>(*lattice) << endl;
            pcout << "Time spent during previous iterations: "
                  << global::timer("iteLog").stop()
                  << ". Current time: " << getTime() << endl;
            global::timer("iteLog").restart(); // 重启log显示之间的循环计时器
        }
    }
    pcout << "Iteration finished." << endl;
    pcout << "Total time spent during iterations: "
          << global::timer("mainLoop").stop() << endl;
    pcout << "Current time: " << getTime();
}
