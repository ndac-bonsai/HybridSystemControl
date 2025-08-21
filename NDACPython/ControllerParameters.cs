using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using System.Xml.Serialization;

namespace NDACPython
{

    /// <summary>
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Source)]
    [Description("Parameters for an Online SLDS model")]
    public class ControllerParameters
    {
        private Dictionary<object,object> _ssParams;

        private float[] _Q;
        private float[] _R;
        private int _N;
        private double _lb;
        private double _ub;
       
        private string SSParamsString;
        private string QString;
        private string RString;
        private string NString;
        private string LBString;
        private string UBString;


        /// <summary>
        /// Finite time horizon in number of bins
        /// </summary>
        [JsonProperty("N")]
        [Description("Finite time horizon in number of bins")]
        public int N
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
        /// State-space parameters of the model
        /// </summary>
        [JsonProperty("ssParams")]
        [Description("State-space parameters of the model")]
        [XmlIgnore]
        public Dictionary<object,object> ssParams
        {
            get
            {
                return _ssParams;
            }
            set
            {
                _ssParams = value;
                SSParamsString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// State error penalty
        /// </summary>
        [JsonProperty("Q")]
        [Description("State error penalty")]
        public float[] Q
        {
            get
            {
                return _Q;
            }
            set
            {
                _Q = value;
                QString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// Input cost matrix
        /// </summary>
        [JsonProperty("R")]
        [Description("Input cost Matrix")]
        public float[] R
        {
            get
            {
                return _R;
            }
            set
            {
                _R = value;
                RString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// Lower bound on inputs
        /// </summary>
        [JsonProperty("lb")]
        [Description("Lower bound on Inputs")]
        public double lb
        {
            get
            {
                return _lb;
            }
            set
            {
                _lb = value;
                LBString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// Upper bound on inputs
        /// </summary>
        [JsonProperty("ub")]
        [Description("Upper bound on inputs")]
        public double ub
        {
            get
            {
                return _ub;
            }
            set
            {
                _ub = value;
                UBString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ControllerParameters"/> class.
        /// </summary>
        public ControllerParameters()
        {
            _ssParams = new Dictionary<object, object>();
            _Q = new float[] { 10f, 10f };
            _R = new float[] { 5f, 5f };
            _N = 10;
            _lb = 0.0F;
            _ub = 1.0F;
        }

        /// <summary>
        /// Generates parameters for a SLDS Model
        /// </summary>
        public IObservable<ControllerParameters> Process()
        {
            return Observable.Defer(() => Observable.Return(
                new ControllerParameters
                {
                    ssParams = _ssParams,
                    Q = _Q,
                    R = _R,
                    N = _N,
                    lb = _lb,
                    ub = _ub,
                }));
        }

        /// <summary>
        /// Gets the model parameters from a PyObject of a SLDS Model
        /// </summary>
        /// May not need this
        /*
        public IObservable<ControllerParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var SSParamsPyObj = (Dictionary<object, object>)pyObject.GetArrayAttr("ssParams");
                var alphaVariancePyObj = (double)pyObject.GetArrayAttr("alpha_variance");
                var xVariancePyObj = (double)pyObject.GetArrayAttr("x_variance");
                var NPyObj = pyObject.GetAttr<int>("N");
                
                var transitionsPyObj = pyObject.GetAttr<string>("transitions");
                var dynamicsPyObj = pyObject.GetAttr<string>("dynamics");
                var emissionsPyObj = pyObject.GetAttr<string>("emissions");
                var dtPyObj = pyObject.GetAttr<float>("dt");
                var alphaVariancePyObj = (double)pyObject.GetArrayAttr("alpha_variance");
                var xVariancePyObj = (double)pyObject.GetArrayAttr("x_variance");
                var sigmasScalerPyObj = pyObject.GetAttr<float>("SigmasScaler");
                var WithNoisePyObj = pyObject.GetAttr<string>("with_noise");

                return new SLDSParameters
                {
                    N = NPyObj,
                    D = DPyObj,
                    K = KPyObj,
                    M = MPyObj,
                    transitions = transitionsPyObj,
                    dynamics = dynamicsPyObj,
                    emissions = emissionsPyObj,
                    dt = dtPyObj,
                    alpha_variance = alphaVariancePyObj,
                    x_variance = xVariancePyObj,
                    sigmasScaler = sigmasScalerPyObj,
                    with_noise = WithNoisePyObj
                };
            });
        }
        */

        /// <summary>
        /// Generates parameters for a Controller on each input
        /// </summary>
        public IObservable<ControllerParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new ControllerParameters
                {
                    ssParams = _ssParams,
                    Q = _Q,
                    R = _R,
                    N = _N,
                    lb = _lb,
                    ub = _ub
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ssParams={SSParamsString},Q={QString},R={RString},N={NString},lb={LBString},ub={UBString}";
        }
    }

}