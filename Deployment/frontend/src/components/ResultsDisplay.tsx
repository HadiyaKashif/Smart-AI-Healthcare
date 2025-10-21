import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  MapPin,
  Mail,
} from "lucide-react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown"; 

const getRiskConfig = (risk) => {
  switch (risk?.toLowerCase()) {
    case "good":
      return {
        color: "green-500",
        text: "Low risk. Maintain your healthy lifestyle!",
        icon: CheckCircle2,
        gradient: "bg-gradient-to-br from-green-400 to-green-600",
      };
    case "moderate":
    case "moderate risk":
      return {
        color: "yellow-500",
        text: "Moderate risk detected. Some improvements are needed.",
        icon: AlertTriangle,
        gradient: "bg-gradient-to-br from-yellow-400 to-yellow-600",
      };
    case "bad":
    case "high":
      return {
        color: "red-500",
        text: "High risk detected. Medical attention recommended.",
        icon: AlertCircle,
        gradient: "bg-gradient-to-br from-red-400 to-red-600",
      };
    default:
      return {
        color: "gray-400",
        text: "Risk level unavailable.",
        icon: AlertCircle,
        gradient: "bg-gradient-to-br from-gray-400 to-gray-600",
      };
  }
};
const ResultsDisplay = ({ results, onReset }) => {
  const { risk, explanation, diagnosis, nextSteps, doctors, name } = results || {};
  const config = getRiskConfig(risk);
  const RiskIcon = config.icon;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header with Patient Name */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="p-6 shadow-soft">
          <h1 className="text-2xl font-bold text-center mb-2">
            Health Report for {name || "Patient"}
          </h1>
          <p className="text-muted-foreground text-center">
            Personalized health assessment based on your vital signs
          </p>
        </Card>
      </motion.div>

      {/* Risk Assessment Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card
          className={`p-8 shadow-glow border-2 border-${config.color} transition-all`}
        >
          <div className="flex items-start gap-6">
            <div
              className={`${config.gradient} w-20 h-20 rounded-full flex items-center justify-center flex-shrink-0`}
            >
              <RiskIcon className="w-10 h-10 text-white" />
            </div>
            <div className="flex-1">
              <h2 className="text-3xl font-bold mb-2 capitalize">
                {risk || "Unknown Risk"}
              </h2>   
              <p className="text-muted-foreground text-lg">{config.text}</p>
            </div>
          </div>
        </Card>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Explanation */}
        {explanation && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 h-full shadow-soft">
              <div className="flex items-center gap-2 mb-4">
                <AlertCircle className="w-6 h-6 text-primary" />
                <h3 className="text-xl font-semibold">Results Explanation</h3>
              </div>
              <div className="prose prose-sm max-w-none whitespace-pre-line text-muted-foreground">
                <ReactMarkdown>{explanation}</ReactMarkdown>
              </div>
            </Card>
          </motion.div>
        )}

        {/* Diagnosis */}
        {diagnosis && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card className="p-6 h-full shadow-soft">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle className="w-6 h-6 text-secondary" />
                <h3 className="text-xl font-semibold">Possible Diagnosis</h3>
              </div>
              <div className="prose prose-sm max-w-none whitespace-pre-line text-muted-foreground">
                <ReactMarkdown>{diagnosis}</ReactMarkdown>
              </div>
            </Card>
          </motion.div>
        )}
      </div>

      {/* Next Steps */}
      {nextSteps?.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className="p-6 shadow-soft">
            <div className="flex items-center gap-2 mb-4">
              <Lightbulb className="w-6 h-6 text-green-600" />
              <h3 className="text-xl font-semibold">Recommended Next Steps</h3>
            </div>
            <div className="space-y-3">
              {nextSteps.map((step, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                  <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <div className="prose prose-sm max-w-none whitespace-pre-line">
                    <ReactMarkdown>{step}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      )}

      {/* Doctors Section */}
      {doctors?.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card className="p-6 shadow-soft">
            <h3 className="text-xl font-semibold mb-4">
              Nearby Healthcare Professionals
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              {doctors.map((doctor, index) => (
                <Card
                  key={index}
                  className="p-4 border-2 border-border hover:border-primary transition-colors"
                >
                  <h4 className="font-semibold text-lg mb-2">{doctor.name}</h4>
                  <p className="text-sm text-muted-foreground mb-1">
                    {doctor.specialty}
                  </p>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                    <MapPin className="w-4 h-4" />
                    <span>{doctor.distance_km} km away</span>
                  </div>
                  {doctor.email && doctor.email !== "N/A" && (
                    <a
                      href={`mailto:${doctor.email}`}
                      className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
                    >
                      <Mail className="w-4 h-4" />
                      Contact
                    </a>
                  )}
                </Card>
              ))}
            </div>
          </Card>
        </motion.div>
      )}

      {/* Reset Button */}
      <div className="flex justify-center gap-4 pt-6">
        <Button
          onClick={onReset}
          size="lg"
          className="bg-gradient-primary hover:opacity-90 text-white px-8"
        >
          Analyze Again
        </Button>
      </div>
    </div>
  );
};
export default ResultsDisplay;
