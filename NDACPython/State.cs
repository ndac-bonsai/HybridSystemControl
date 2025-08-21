using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;

namespace NDACPython
{
    // <summary>
    /// State of a SLDS (continuous states x and regime z)
    /// </summary>
    [Description("State of a SLDS (continuous states x and regime z)")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class State
    {

        private double[] _x;

        private int _z;


        /// <summary>
        /// continuous state vector - d x 1 dimensional matrix where d is number of states
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("x")]
        [Description("Continuous states")]
        public double[] X
        {
            get
            {
                return _x;
            }
            set
            {
                _x = value;
            }
        }

        /// <summary>
        /// discrete regime
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("P")]
        [Description("Discrete Regime")]
        public int Z
        {
            get
            {
                return _z;
            }
            set
            {
                _z = value;
            }
        }

        /// <summary>
        /// Grabs the state of a SLDS from a type of PyObject
        /// /// </summary>
        public IObservable<State> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var xPyObj = (double[])pyObject.GetArrayAttr("x");
                var zPyObj = (int)pyObject.GetArrayAttr("z");

                return new State
                {
                    X = xPyObj,
                    Z = zPyObj
                };
            });
        }
    }
}
