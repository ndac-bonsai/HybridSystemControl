using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;
using Bonsai;
using System.Collections.Generic;

namespace NDACPython
{
    // <summary>
    /// State of a SLDS (continuous states x and regime z)
    /// </summary>
    [Description("State-space parameters needed from the model to create a Controller")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class SSParameters
    {

        private Dictionary<object,object> _SSparams;
        private string SSParamsString;

        /// <summary>
        /// model parameters
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("SSparams")]
        [Description("Dictionary of state-space parameters of the model")]
        public Dictionary<object,object> SSParams
        {
            get
            {
                return _SSparams;
            }
            set
            {
                _SSparams = value;
                SSParamsString = StringFormatter.FormatToPython(value);
            }
        }

        /// <summary>
        /// Grabs the state of a SLDS from a type of PyObject
        /// /// </summary>
        public IObservable<SSParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var SSParamsPyObj = (Dictionary<object, object>)pyObject.GetArrayAttr("ssParams");

                return new SSParameters
                {
                    SSParams = SSParamsPyObj
                };
            });
        }
        public override string ToString()
        {
            return $"controller_params={SSParamsString}";
        }
    }
}
