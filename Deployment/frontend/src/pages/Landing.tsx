import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Heart, Activity, Stethoscope, Shield } from "lucide-react";

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/5 to-secondary/10">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="max-w-4xl mx-auto text-center">
          {/* Icon */}
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-primary mb-8 animate-pulse-glow">
            <Heart className="w-10 h-10 text-white" />
          </div>

          {/* Title */}
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent animate-fade-in">
            Smart Health Alert System 🩺
          </h1>

          {/* Tagline */}
          <p className="text-2xl md:text-3xl text-foreground/80 mb-8 animate-fade-in" style={{ animationDelay: "0.1s" }}>
            Predict your health risks and get AI-powered recommendations
          </p>

          {/* Description */}
          <p className="text-lg text-muted-foreground mb-12 max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: "0.2s" }}>
            Our advanced AI system analyzes your vital signs to detect anomalies, 
            generates intelligent health insights, and suggests nearby healthcare 
            professionals when needed.
          </p>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12 animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <div className="bg-card p-6 rounded-xl shadow-soft hover:shadow-glow transition-all duration-300">
              <Activity className="w-8 h-8 text-primary mx-auto mb-4" />
              <h3 className="font-semibold mb-2">Detect Anomalies</h3>
              <p className="text-sm text-muted-foreground">
                Advanced algorithms analyze your vitals to identify potential health issues
              </p>
            </div>

            <div className="bg-card p-6 rounded-xl shadow-soft hover:shadow-glow transition-all duration-300">
              <Shield className="w-8 h-8 text-secondary mx-auto mb-4" />
              <h3 className="font-semibold mb-2">AI-Driven Insights</h3>
              <p className="text-sm text-muted-foreground">
                Get personalized causes and solutions powered by cutting-edge AI
              </p>
            </div>

            <div className="bg-card p-6 rounded-xl shadow-soft hover:shadow-glow transition-all duration-300">
              <Stethoscope className="w-8 h-8 text-success mx-auto mb-4" />
              <h3 className="font-semibold mb-2">Find Doctors</h3>
              <p className="text-sm text-muted-foreground">
                Instant access to nearby healthcare professionals when you need them
              </p>
            </div>
          </div>

          {/* CTA Button */}
          <Button
            size="lg"
            onClick={() => navigate("/analyze")}
            className="bg-gradient-primary hover:opacity-90 text-white px-8 py-6 text-lg shadow-glow animate-fade-in-scale"
            style={{ animationDelay: "0.4s" }}
          >
            Analyze Your Health →
          </Button>
        </div>
      </div>

      {/* Footer */}
      <div className="container mx-auto px-4 pb-8 text-center text-sm text-muted-foreground">
        <p>Powered by AI • Secure • Confidential</p>
      </div>
    </div>
  );
};

export default Landing;
