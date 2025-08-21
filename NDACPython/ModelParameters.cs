using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;
using System.Collections.Generic;
using System.Data.SqlTypes;

namespace NDACPython
{
    // <summary>
    /// Dynamics, Transitions, and Emissions Parameters of SLDS Model
    /// </summary>
    [Description("Dynamics, Transitions, and Emissions Parameters of SLDS Model")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ModelParameters
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
        private double _x_variance;
        private bool _constant;

        /// <summary>
        /// Number of observations
        /// </summary>
        [JsonProperty("N")]
        [Description("Number of observations")]
        public int N
        {
            get
            {
                return _N;
            }
            set
            {
                _N = value;
            }
        }

        /// <summary>
        /// Number of Regimes
        /// </summary>
        [JsonProperty("K")]
        [Description("Number of Regimes")]
        public int K
        {
            get
            {
                return _K;
            }
            set
            {
                _K = value;
            }
        }

        /// <summary>
        /// Number of continuous states
        /// </summary>
        [JsonProperty("D")]
        [Description("Number of continuous states")]
        public int D
        {
            get
            {
                return _D;
            }
            set
            {
                _D = value;
            }
        }

        /// <summary>
        /// Number of inputs
        /// </summary>
        [JsonProperty("M")]
        [Description("Number of inputs")]
        public int M
        {
            get
            {
                return _M;
            }
            set
            {
                _M = value;
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
            }
        }

        public double x_variance
        {
            get
            {
                return _x_variance;
            }
            set
            {
                _x_variance = value;
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
            }
        }

        /// <summary>
        /// Grabs the state of a SLDS from a type of PyObject
        /// /// </summary>
        //

        public IObservable<ModelParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                /*
                var AsPyObj = (Array)pyObject.GetArrayAttr("As");
                var BsPyObj = (Array)pyObject.GetArrayAttr("Bs");
                var CsPyObj = (Array)pyObject.GetArrayAttr("Cs");
                var DsPyObj = (Array)pyObject.GetArrayAttr("Ds");
                var dsPyObj = (double[])pyObject.GetArrayAttr("ds");
                var sigmaSqInitPyObj = (Array)pyObject.GetArrayAttr("Sigmas_init");
                var QsPyObj = (Array)pyObject.GetArrayAttr("Sigmas");
                var RsPyObj = (double[])pyObject.GetArrayAttr("Rs");
                var rPyObj = (double[])pyObject.GetArrayAttr("r");
                var pi0PyObj = (double[])pyObject.GetArrayAttr("pi0");
                var dtPyObj = pyObject.GetAttr<double>("dt");
                var alphaVariancePyObj = pyObject.GetAttr<double>("alpha_variance");
                var xVariancePyObj = pyObject.GetAttr<double>("x_variance");
                var ConstantPyObj = pyObject.GetAttr<bool>("constant");
                */
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
                var alphaVariancePyObj = pyObject.GetAttr<double>("alpha_variance");
                var xVariancePyObj = pyObject.GetAttr<double>("x_variance");
                var ConstantPyObj = pyObject.GetAttr<bool>("constant");
                return new ModelParameters
                {
                    N = NPyObj,
                    D = DPyObj,
                    K = KPyObj,
                    M = MPyObj,
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
    }
}
