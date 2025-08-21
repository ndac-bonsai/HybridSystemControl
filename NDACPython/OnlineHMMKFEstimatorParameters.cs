using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;
using System.Collections.Generic;

namespace NDACPython
{

    /// <summary>
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Source)]
    [Description("Parameters for an Online SLDS model")]
    public class HMMKFParameters
    {
        private int _N;
        private int _K;
        private int _D;
        private int _M;
        private double[] _As;
        private double[] _Bs;
        private double[] _Cs;
        private double[] _Fs;
        private double[] _ds;
        private double[] _sigmasq_init;
        private double[] _Qs;
        private double[] _Rs;
        private double[] _r;
        private double[] _pi0;
        private double _dt;
        private double _alpha_variance;
        private double[] _x_variance;
        private bool _constant;

        private string NString;
        private string KString;
        private string DString;
        private string MString;
        private string AsString;
        private string BsString;
        private string CsString;
        private string FsString;
        private string dsString;
        private string SigmasSqInitString;
        private string QsString;
        private string RsString;
        private string rString;
        private string pi0String;
        private string dtString;
        private string alphaVarianceString;
        private string xVarianceString;
        private string ConstantString;

        /// <summary>
        /// Number of observations
        /// </summary>
        [JsonProperty("N")]
        [Description("Number of observations")]
        public int N_
        {
            get
            {
                return _N;
            }
            set
            {
                _N = value;
                NString = double.IsNaN(_N) ? "None" : _N.ToString();
            }
        }

        /// <summary>
        /// Number of Regimes
        /// </summary>
        [JsonProperty("K")]
        [Description("Number of Regimes")]
        public int K_
        {
            get
            {
                return _K;
            }
            set
            {
                _K = value;
                KString = double.IsNaN(_K) ? "None" : _K.ToString();
            }
        }

        /// <summary>
        /// Number of continuous states
        /// </summary>
        [JsonProperty("D")]
        [Description("Number of continuous states")]
        public int D_
        {
            get
            {
                return _D;
            }
            set
            {
                _D = value;
                DString = double.IsNaN(_D) ? "None" : _D.ToString();
            }
        }

        /// <summary>
        /// Number of inputs
        /// </summary>
        [JsonProperty("M")]
        [Description("Number of inputs")]
        public int M_
        {
            get
            {
                return _M;
            }
            set
            {
                _M = value;
                MString = double.IsNaN(_M) ? "None" : _M.ToString();
            }
        }
        /// <summary>
        /// State Transition Matrices
        /// </summary>
        [JsonProperty("As")]
        [Description("State Transition Matrices")]
        public double[] As
        {
            get
            {
                return _As;
            }
            set
            {
                _As = value;
                AsString = StringFormatter.FormatToPython(_As);
            }
        }

        /// <summary>
        /// Input-State Matrices
        /// </summary>
        [JsonProperty("Bs")]
        [Description("Input-State Matrices")]
        public double[] Bs
        {
            get
            {
                return _Bs;
            }
            set
            {
                _Bs = value;
                BsString = StringFormatter.FormatToPython(_Bs);
            }
        }

        /// <summary>
        /// State Emissions Matrices
        /// </summary>
        [JsonProperty("Cs")]
        [Description("State Emissions Matrices")]
        public double[] Cs
        {
            get
            {
                return _Cs;
            }
            set
            {
                _Cs = value;
                CsString = StringFormatter.FormatToPython(_Cs);
            }
        }

        /// <summary>
        /// Feed-through emissions Matrices
        /// </summary>
        [JsonProperty("Fs")]
        [Description("Feed-through emissions Matrices")]
        public double[] Fs
        {
            get
            {
                return _Fs;
            }
            set
            {
                _Fs = value;
                FsString = StringFormatter.FormatToPython(_Fs);
            }
        }

        /// <summary>
        /// Emissions bias Matrices
        /// </summary>
        [JsonProperty("ds")]
        [Description("Emissions bias Matrices")]
        public double[] ds
        {
            get
            {
                return _ds;
            }
            set
            {
                _ds = value;
                dsString = StringFormatter.FormatToPython(_ds);
            }
        }

        /// <summary>
        /// Initial state variance
        /// </summary>
        [JsonProperty("sigmassq_init")]
        [Description("Initial state variance")]
        public double[] sigmasq_init
        {
            get
            {
                return _sigmasq_init;
            }
            set
            {
                _sigmasq_init = value;
                SigmasSqInitString = StringFormatter.FormatToPython(_sigmasq_init);
            }
        }

        /// <summary>
        /// State Variances
        /// </summary>
        [JsonProperty("Qs")]
        [Description("State Variances")]
        public double[] Qs
        {
            get
            {
                return _Qs;
            }
            set
            {
                _Qs = value;
                QsString = StringFormatter.FormatToPython(_Qs);
            }
        }

        /// <summary>
        /// Regime Transitions Matrices
        /// </summary>
        [JsonProperty("Rs")]
        [Description("Regime Transitions Matrices")]
        public double[] Rs
        {
            get
            {
                return _Rs;
            }
            set
            {
                _Rs = value;
                RsString = StringFormatter.FormatToPython(_Rs);
            }
        }

        /// <summary>
        /// Transitions bias matrix
        /// </summary>
        [JsonProperty("r")]
        [Description("Transitions bias matrix")]
        public double[] r
        {
            get
            {
                return _r;
            }
            set
            {
                _r = value;
                rString = StringFormatter.FormatToPython(_r); ;
            }
        }

        /// <summary>
        /// Timestep of SLDS
        /// </summary>
        [JsonProperty("dt")]
        [Description("Timestep of SLDS")]
        public double dt
        {
            get
            {
                return _dt;
            }
            set
            {
                _dt = value;
                dtString = _dt.ToString();
            }
        }

        public double[] pi0
        {
            get
            {
                return _pi0;
            }
            set
            {
                _pi0 = value;
                pi0String = StringFormatter.FormatToPython(pi0);
            }
        }

        public double alpha_variance
        {
            get
            {
                return _alpha_variance;
            }
            set
            {
                _alpha_variance = value;
                alphaVarianceString = StringFormatter.FormatToPython(value);
            }
        }

        public double[] x_variance
        {
            get
            {
                return _x_variance;
            }
            set
            {
                _x_variance = value;
                xVarianceString = StringFormatter.FormatToPython(value);
            }
        }

        public bool constant
        {
            get
            {
                return _constant;
            }
            set
            {
                _constant = value;
                ConstantString = value.ToString();
            }
        }
        /// <summary>
        /// Initializes a new instance of the <see cref="HMMKFParameters"/> class.
        /// </summary>
        /// 
        /*
        public HMMKFParameters()
        {
            N_ = 5;
            D_ = 2;
            K_ = 3;
            M_ = 2;
            As = new double[K_*D_*D_];
            Bs = new double[K_*D_*M_];
            Cs = new double[1*N_*D_];
            Fs = new double[1*N_*M_];
            ds = new double[1 * N_];
            sigmasq_init = new double[K_ * D_ * D_];
            Qs = new double[K_ * D_ * D_];
            Rs = new double[K_ * D_];
            r = new double[K_];
            pi0 = new double[K_];
            dt = 0.01F;
            alpha_variance = 10000F;
            x_variance = .0005F;
            constant = false;

        }
        */

        /// <summary>
        /// Generates parameters for a SLDS Model
        /// </summary>
        public IObservable<HMMKFParameters> Process()
        {
            return Observable.Defer(() => Observable.Return(
                new HMMKFParameters
                {
                    N_ = _N,
                    D_ = _D,
                    K_ = _K,
                    M_ = _M,
                    As = _As,
                    Bs = _Bs,
                    Cs = _Cs,
                    Fs = _Fs,
                    ds = _ds,
                    sigmasq_init = _sigmasq_init,
                    Qs = _Qs,
                    Rs = _Rs,
                    r = _r,
                    pi0 = _pi0,
                    dt = _dt,
                    alpha_variance = _alpha_variance,
                    x_variance = _x_variance,
                    constant = _constant
                }));
        }

        /// <summary>
        /// Gets the model parameters from a PyObject of a SLDS Model
        /// </summary>
        public IObservable<HMMKFParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var NPyObj = pyObject.GetAttr<int>("N");
                var DPyObj = pyObject.GetAttr<int>("D");
                var KPyObj = pyObject.GetAttr<int>("K");
                var MPyObj = pyObject.GetAttr<int>("M");
                var AsPyObj = (double[])pyObject.GetArrayAttr("As");
                var BsPyObj = (double[])pyObject.GetArrayAttr("Bs");
                var CsPyObj = (double[])pyObject.GetArrayAttr("Cs");
                var FsPyObj = (double[])pyObject.GetArrayAttr("Fs");
                var dsPyObj = (double[])pyObject.GetArrayAttr("ds");
                var sigmaSqInitPyObj = (double[])pyObject.GetArrayAttr("Sigmas_init");
                var QsPyObj = (double[])pyObject.GetArrayAttr("Sigmas");
                var RsPyObj = (double[])pyObject.GetArrayAttr("Rs");
                var rPyObj = (double[])pyObject.GetArrayAttr("r");
                var pi0PyObj = (double[])pyObject.GetArrayAttr("pi0");
                var dtPyObj = pyObject.GetAttr<double>("dt");
                var alphaVariancePyObj = (double)pyObject.GetArrayAttr("alpha_variance");
                var xVariancePyObj = (double[])pyObject.GetArrayAttr("x_variance");
                var ConstantPyObj = pyObject.GetAttr<bool>("constant");

                return new HMMKFParameters
                {
                    N_ = NPyObj,
                    D_ = DPyObj,
                    K_ = KPyObj,
                    M_ = MPyObj,
                    As = AsPyObj,
                    Bs = BsPyObj,
                    Cs = CsPyObj,
                    Fs = FsPyObj,
                    ds = dsPyObj,
                    sigmasq_init = sigmaSqInitPyObj,
                    Qs = QsPyObj,
                    Rs = RsPyObj,
                    r = rPyObj,
                    pi0 = pi0PyObj,
                    dt = dtPyObj,
                    alpha_variance = alphaVariancePyObj,
                    x_variance = xVariancePyObj,
                    constant = ConstantPyObj
                };
            });
        }

        /// <summary>
        /// Generates parameters for an SLDS Model on each input
        /// </summary>
        public IObservable<HMMKFParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new HMMKFParameters
                {
                    N_ = _N,
                    D_ = _D,
                    K_ = _K,
                    M_ = _M,
                    As = _As,
                    Bs = _Bs,
                    Cs = _Cs,
                    Fs = _Fs,
                    ds = _ds,
                    sigmasq_init = _sigmasq_init,
                    Qs = _Qs,
                    Rs = _Rs,
                    r = _r,
                    pi0 = _pi0,
                    dt = _dt,
                    alpha_variance = _alpha_variance,
                    x_variance = _x_variance,
                    constant = _constant
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"N={NString},D={DString},K={KString},M={MString},As={AsString},Bs={BsString},Cs={CsString},Fs={FsString},ds={dsString},sigmasq_init={SigmasSqInitString},Qs={QsString},Rs={RsString},r={rString},pi0={pi0String},alpha_variance={alphaVarianceString},x_variance={xVarianceString},dt={dtString},constant={ConstantString}";
        }
    }

}