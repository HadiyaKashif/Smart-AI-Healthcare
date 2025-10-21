import { useState } from "react";
import HealthForm from "@/components/HealthForm";
import PipelineAnimation from "@/components/PipelineAnimation";
import ResultsDisplay from "@/components/ResultsDisplay";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const Analyze = () => {
  const navigate = useNavigate();
  const [stage, setStage] = useState("form"); // form, processing, results
  const [results, setResults] = useState(null);
const handleFormSubmit = async (formData) => {
  setStage("processing");

  const formattedData = {
    "Name": formData.name,
    "Gender": formData.gender,
    "Age": formData.age,
    "Systolic BP": formData.systolicBP,
    "Diastolic BP": formData.diastolicBP,
    "Cholesterol": formData.cholesterol,
    "BMI": formData.bmi,
    "Smoker": formData.smoker,
    "Diabetes": formData.diabetes,
    "Email": formData.email,
    "Latitude": formData.latitude,
    "Longitude": formData.longitude,
  };

  try {
    console.log(JSON.stringify(formattedData));
    
    const response = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formattedData),
    });

    const data = await response.json();
    
    // Updated destructuring to match new response structure
    const { risk, explanation, diagnosis, nextSteps, hospitals, name } = data;

    // Simulate processing time for animation
    setTimeout(() => {
      setResults({ 
        risk,
        explanation, 
        diagnosis,
        nextSteps,
        hospitals,
        name 
      });
      setStage("results");
    }, 5000);
  } catch (error) {
    console.error("Error submitting form:", error);
    
    // Mock data for development with new structure
    setTimeout(() => {
      setResults({
        name: "Test Patient",
        risk: "Some Risk",
        explanation: "Based on your recent readings, we've categorized your risk as Moderate to High.",
        diagnosis: "Your pattern of vitals is commonly associated with Primary Hypertension.",
        nextSteps: [
          "Schedule a Follow-Up Appointment",
          "Monitor at Home",
          "Lifestyle Considerations"
        ],
        doctors: [
          {
            name: "Fallback doctor name 1",
            specialty: "General Practitioner",
            distance_km: 3.2,
            email: "ahmed@example.com"
          },
        ]
      });
      setStage("results");
    }, 5000);
  }
};
  const handleReset = () => {
    setStage("form");
    setResults(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/5 to-secondary/10">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="mr-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <h1 className="text-3xl font-bold">Health Analysis</h1>
        </div>

        {/* Content */}
        <div className="max-w-6xl mx-auto">
          {stage === "form" && (
            <HealthForm onSubmit={handleFormSubmit} />
          )}

          {stage === "processing" && (
            <PipelineAnimation />
          )}

          {stage === "results" && results && (
            <ResultsDisplay results={results} onReset={handleReset} />
          )}
        </div>
      </div>
    </div>
  );
};

export default Analyze;
