import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Brain, Activity, UserCheck, CheckCircle2 } from "lucide-react";
import { motion } from "framer-motion";

const PipelineAnimation = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      icon: Activity,
      title: "Model Prediction",
      description: "Analyzing your vital signs...",
      color: "text-primary",
    },
    {
      icon: Brain,
      title: "RAG Reasoning",
      description: "Generating AI insights...",
      color: "text-secondary",
    },
    {
      icon: UserCheck,
      title: "Doctor Suggestions",
      description: "Finding nearby specialists...",
      color: "text-success",
    },
  ];

useEffect(() => {
  const totalDuration = 10000; // 10 seconds total
  const stepDuration = totalDuration / steps.length; // ≈3333 ms per step

  const interval = setInterval(() => {
    setActiveStep((prev) => {
      if (prev < steps.length - 1) return prev + 1;
      clearInterval(interval);
      return prev;
    });
  }, stepDuration);

  return () => clearInterval(interval);
}, []);

  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      <Card className="p-8 shadow-soft">
        <div className="text-center mb-12">
          <h2 className="text-2xl font-bold mb-2">Processing Your Health Data</h2>
          <p className="text-muted-foreground">AI pipeline in action</p>
        </div>

        {/* Pipeline Flow */}
        <div className="space-y-8">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === activeStep;
            const isCompleted = index < activeStep;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.5 }}
                className="relative"
              >
                {/* Connection Line */}
                {index < steps.length - 1 && (
                  <div className="absolute left-8 top-16 w-0.5 h-16 bg-border">
                    {isCompleted && (
                      <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: "100%" }}
                        transition={{ duration: 2 }}
                        className="w-full bg-gradient-primary"
                      />
                    )}
                  </div>
                )}

                {/* Step Card */}
                <div
                  className={`flex items-start gap-4 p-6 rounded-lg border-2 transition-all duration-300 ${
                    isActive
                      ? "border-primary bg-primary/5 shadow-glow"
                      : isCompleted
                      ? "border-success bg-success/5"
                      : "border-border bg-card"
                  }`}
                >
                  {/* Icon */}
                  <div
                    className={`flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center ${
                      isActive
                        ? "bg-gradient-primary animate-pulse-glow"
                        : isCompleted
                        ? "bg-gradient-success"
                        : "bg-muted"
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="w-8 h-8 text-white" />
                    ) : (
                      <Icon className={`w-8 h-8 ${isActive ? "text-white" : "text-muted-foreground"}`} />
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <h3 className={`text-xl font-semibold mb-1 ${isActive ? "text-primary" : ""}`}>
                      {step.title}
                    </h3>
                    <p className="text-muted-foreground">{step.description}</p>

                    {/* Loading Spinner */}
                    {isActive && (
                      <motion.div
                        className="mt-4 flex gap-2"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        {[0, 1, 2].map((i) => (
                          <motion.div
                            key={i}
                            className="w-2 h-2 rounded-full bg-primary"
                            animate={{
                              y: [0, -10, 0],
                            }}
                            transition={{
                              duration: 0.6,
                              repeat: Infinity,
                              delay: i * 0.2,
                            }}
                          />
                        ))}
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Processing Message */}
        <div className="text-center mt-12">
          <motion.p
            className="text-muted-foreground"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Please wait while we analyze your health data...
          </motion.p>
        </div>
      </Card>
    </div>
  );
};

export default PipelineAnimation;
