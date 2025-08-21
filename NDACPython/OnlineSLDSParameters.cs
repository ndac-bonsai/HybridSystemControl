﻿using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Collections.Generic;

namespace NDACPython
{

    /// <summary>
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Source)]
    [Description("Parameters for an Online SLDS model")]
    public class SLDSParameters
    {

        private int _N;
        private int _K;
        private int _D;
        private int _M;
        private string _transitions;
        private string _dynamics;
        private string _emissions;
        //private Dictionary<transitions_kwkeys, transition_kwvals>;
        //private Dictionary emissions_kwargs;
        //private Dictionary dynamics_kwargs;
        private float _dt;
        private double _alpha_variance;
        private double _x_variance;
        private double _sigmasScaler;
        private string _with_noise;

        private string NString;
        private string KString;
        private string DString;
        private string MString;
        private string transitionsString;
        private string dynamicsString;
        private string emissionsString;
        private string dtString;
        private string alphaVarianceString;
        private string xVarianceString;
        private string sigmasScalerString;
        private string WithNoiseString;

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
                NString = double.IsNaN(_N) ? "None" : _N.ToString();
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
                KString = double.IsNaN(_K) ? "None" : _K.ToString();
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
                DString = double.IsNaN(D) ? "None" : _D.ToString();
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
                MString = double.IsNaN(M) ? "None" : M.ToString();
            }
        }
    
        /// <summary>
        /// Transition class of LDS.
        /// </summary>
        [JsonProperty("transitions")]
        [Description("Transition class of SLDS")]
        public string transitions
        {
            get
            {
                return _transitions;
            }
            set
            {
                _transitions = value;
                transitionsString = value;
            }
        }
    
        /// <summary>
        /// Dynamics class of SLDS
        /// </summary>
        [JsonProperty("dynamics")]
        [Description("Dynamics class of SLDS")]
        public string dynamics
        {
            get
            {
                return _dynamics;
            }
            set
            {
                _dynamics = value;
                dynamicsString = value;
            }
        }
    
        /// <summary>
        /// Emissions class of SLDS
        /// </summary>
        [JsonProperty("emissions")]
        [Description("Emissions class of SLDS")]
        public string emissions
        {
            get
            {
                return _emissions;
            }
            set
            {
                _emissions = value;
                emissionsString = value;
            }
        }
    
        /// <summary>
        /// Timestep of SLDS
        /// </summary>
        [JsonProperty("dt")]
        [Description("Timestep of LDS")]
        public float dt
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
        /// <summary>
        /// variance in innovation
        /// </summary>
        [JsonProperty("alpha_variance")]
        [Description("variance in innovation")]
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
        /// <summary>
        /// variance in estimate
        /// </summary>
        [JsonProperty("x_variance")]
        [Description("variance in estimate")]
        public double x_variance
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
        /// <summary>
        /// scaler on model variances
        /// </summary>
        [JsonProperty("sigmaScaler")]
        [Description("scaler on model variances")]
        public double sigmasScaler
        {
            get
            {
                return _sigmasScaler;
            }
            set
            {
                _sigmasScaler = value;
                sigmasScalerString = value.ToString();
            }
        }
        /// <summary>
        /// Whether to use noise on dynamics
        /// </summary>
        [JsonProperty("with_noice")]
        [Description("Whether to use noise on dynamics")]
        public string with_noise
        {
            get
            {
                return _with_noise;
            }
            set
            {
                _with_noise = value;
                WithNoiseString = value.ToString();
            }
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="SLDSParameters"/> class.
        /// </summary>
        public SLDSParameters ()
        {
            N = 5;
            D = 2;
            K = 3;
            M = 2;
            transitions = "\"standard\"";
            dynamics = "\"gaussian\"";
            emissions = "\"poisson\"";
            dt = .01F;
            alpha_variance = .001F;
            x_variance = .1F;
            sigmasScaler = 1F;
            with_noise = "True";
        }

        /// <summary>
        /// Generates parameters for a SLDS Model
        /// </summary>
        public IObservable<SLDSParameters> Process()
        {
    		return Observable.Defer(() => Observable.Return(
                new SLDSParameters {
    				N = _N,
                    D = _D,
                    K = _K,
                    M = _M,
                    transitions = transitions,
                    dynamics = dynamics,
                    emissions = emissions,
                    dt = dt,
                    alpha_variance = _alpha_variance,
                    x_variance = _x_variance,
                    sigmasScaler = _sigmasScaler,
                    with_noise = _with_noise
                }));
        }

        /// <summary>
        /// Gets the model parameters from a PyObject of a SLDS Model
        /// </summary>
        public IObservable<SLDSParameters> Process(IObservable<PyObject> source)
        {
    		return Observable.Select(source, pyObject =>
    		{
                var NPyObj = pyObject.GetAttr<int>("N");
                var DPyObj = pyObject.GetAttr<int>("D");
                var KPyObj = pyObject.GetAttr<int>("K");
                var MPyObj = pyObject.GetAttr<int>("M");
                var transitionsPyObj = pyObject.GetAttr<string>("transitions");
                var dynamicsPyObj = pyObject.GetAttr<string>("dynamics");
                var emissionsPyObj = pyObject.GetAttr<string>("emissions");
                var dtPyObj = pyObject.GetAttr<float>("dt");
                var alphaVariancePyObj = (double)pyObject.GetArrayAttr("alpha_variance");
                var xVariancePyObj = (double)pyObject.GetArrayAttr("x_variance");
                var sigmasScalerPyObj = pyObject.GetAttr<float>("SigmasScaler");
                var WithNoisePyObj = pyObject.GetAttr<string>("with_noise");

                return new SLDSParameters {
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
    
        /// <summary>
        /// Generates parameters for an SLDS Model on each input
        /// </summary>
        public IObservable<SLDSParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new SLDSParameters {
                    N = _N,
                    D = _D,
                    K = _K,
                    M = _M,
                    transitions = transitions,
                    dynamics = dynamics,
                    emissions = emissions,
                    dt = dt,
                    alpha_variance = _alpha_variance,
                    x_variance = _x_variance,
                    sigmasScaler = sigmasScaler,
                    with_noise = _with_noise
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"N={NString},D={DString},K={KString},M={MString},transitions={transitionsString},dynamics={dynamicsString},emissions={emissionsString},dt={dtString},alpha_variance={alphaVarianceString},x_variance={xVarianceString},SigmasScaler={sigmasScalerString},with_noise={WithNoiseString}";
        }
    }

}