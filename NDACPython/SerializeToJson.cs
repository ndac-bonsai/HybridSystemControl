using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Newtonsoft.Json;
using Bonsai;

namespace NDACPython
{
    /// <summary>
    /// Serializes a sequence of data model objects into JSON strings.
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [Description("Serializes a sequence of data model objects into JSON strings.")]
    public class SerializeToJson
    {
        private IObservable<string> Process<T>(IObservable<T> source)
        {
            return source.Select(value => JsonConvert.SerializeObject(value));
        }

        /// <summary>
        /// Serializes each <see cref="SLDSModelParameters"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="SLDSModelParameters"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="SLDSModelParameters"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<SLDSParameters> source)
        {
            return Process<SLDSParameters>(source);
        }
        public IObservable<string> Process(IObservable<HMMKFParameters> source)
        {
            return Process<HMMKFParameters>(source);
        }
        public IObservable<string> Process(IObservable<ModelParameters> source)
        {
            return Process<ModelParameters>(source);
        }
        public IObservable<string> Process(IObservable<SSParameters> source)
        {
            return Process<SSParameters>(source);
        }
        public IObservable<string> Process(IObservable<ControllerParameters> source)
        {
            return Process<ControllerParameters>(source);
        }
        /// <summary>
        /// Serializes each <see cref="State"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="State"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="State"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<State> source)
        {
            return Process<State>(source);
        }
        
        public IObservable<string> Process(IObservable<Observation> source)
        {
            return Process<Observation>(source);
        }



        /// <summary>
        /// Serializes each <see cref="State1"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="State1"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="State1"/> object.
        /// </returns>
        //public IObservable<string> Process(IObservable<State1> source)
        // {
        //  return Process<State1>(source);
        // }
    }
}