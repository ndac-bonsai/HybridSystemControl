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
    public class Observation
    {

        private double[] _y;

        /// <summary>
        /// observations - n x 1 dimensional matrix where n is number observations
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("y")]
        [Description("Observations")]
        public double[] Y
        {
            get
            {
                return _y;
            }
            set
            {
                _y = value;
            }
        }

        /// <summary>
        /// Grabs the observations of a SLDS from a type of PyObject
        /// /// </summary>
        public IObservable<Observation> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var yPyObj = (double[])pyObject.GetArrayAttr("y");

                return new Observation
                {
                    Y = yPyObj
                };
            });
        }
    }
}
