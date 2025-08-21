using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;


namespace NDACPython
{
    [Combinator]
    [Description("Converts an incoming stream of Python objects of numpy float32 arrays to C# float arrays")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Py2C
    {
        public IObservable<float[]> Process(IObservable<PyObject> source)
        {


            return source.Select(pythonObject =>
            {
                // Create list to store all floats in Python numpy array
                List<float> list = new List<float>();

                // Begins Python Global Interpreter Lock 
                using (Py.GIL())
                {

                    // Fills a C# list with all values from the Python array
                    if (pythonObject.IsIterable())
                    {
                        var iterator = pythonObject.GetIterator();
                        while (iterator.MoveNext())
                        {

                            var prob = iterator.Current;
                            try
                            {
                                float value = prob.As<float>();
                                list.Add(value);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("Error converting item to double: " + ex.Message);
                                return new float[0];
                            }
                            finally
                            {
                                prob.Dispose();
                            }
                        }

                    }
                    else
                    {
                        Console.WriteLine("Object is not iterable");
                        return new float[0];
                    }
                }

                // Converts the C# list to a float array
                return list.ToArray();

            });

        }
    }
}
