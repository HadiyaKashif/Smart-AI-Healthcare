import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MapPin } from "lucide-react";
import { toast } from "sonner";

const HealthForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    name: "",
    gender: "",
    age: "",
    systolicBP: "",
    diastolicBP: "",
    cholesterol: "",
    bmi: "",
    smoker: false,
    diabetes: false,
    email: "",
    latitude: "",
    longitude: "",
  });

  const handleInputChange = (field, value) => {
    // use functional update to avoid stale state when updating multiple fields
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const validateForm = () => {
    if (
      !formData.gender ||
      !formData.age ||
      !formData.email ||
      !formData.name
    ) {
      toast.error("Please fill in all required fields");
      return false;
    }

    const age = Number(formData.age);
    if (age < 1 || age > 120) {
      toast.error("Please enter a valid age");
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      toast.error("Please enter a valid email address");
      return false;
    }

    const systolic = Number(formData.systolicBP);
    if (formData.systolicBP && (systolic < 70 || systolic > 200)) {
      toast.error("Systolic BP should be between 70 and 200");
      return false;
    }

    const diastolic = Number(formData.diastolicBP);
    if (formData.diastolicBP && (diastolic < 40 || diastolic > 130)) {
      toast.error("Diastolic BP should be between 40 and 130");
      return false;
    }

    return true;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (validateForm()) {
      toast.success("Form submitted! Analyzing your health data...");
      onSubmit(formData);
    }
  };

  const handleLocationClick = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          handleInputChange("latitude", position.coords.latitude.toFixed(6));
          handleInputChange("longitude", position.coords.longitude.toFixed(6));
          toast.success("Location captured successfully!");
        },
        (error) => {
          toast.error(
            "Unable to get location. Please enable location services."
          );
        }
      );
    } else {
      toast.error("Geolocation is not supported by this browser.");
    }
  };

  return (
    <Card className="p-8 shadow-soft animate-fade-in">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold mb-2">
            Enter Your Health Information
          </h2>
          <p className="text-muted-foreground">
            All fields are important for accurate analysis
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Email */}
          <div className="space-y-2 ">
            <Label htmlFor="name">Name *</Label>
            <Input
              id="name"
              placeholder="eg: John Doe"
              value={formData.name}
              onChange={(e) => handleInputChange("name", e.target.value)}
            />
          </div>
          {/* Gender */}
          <div className="space-y-2">
            <Label htmlFor="gender">Gender *</Label>
            <Select
              value={formData.gender}
              onValueChange={(value) => handleInputChange("gender", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Age */}
          <div className="space-y-2">
            <Label htmlFor="age">Age *</Label>
            <Input
              id="age"
              type="number"
              placeholder="Enter your age"
              value={formData.age}
              onChange={(e) => handleInputChange("age", e.target.value)}
              min="1"
              max="120"
            />
          </div>

          {/* Systolic BP */}
          <div className="space-y-2">
            <Label htmlFor="systolicBP">Systolic BP (mmHg)</Label>
            <Input
              id="systolicBP"
              type="number"
              placeholder="e.g., 120"
              value={formData.systolicBP}
              onChange={(e) => handleInputChange("systolicBP", e.target.value)}
            />
          </div>

          {/* Diastolic BP */}
          <div className="space-y-2">
            <Label htmlFor="diastolicBP">Diastolic BP (mmHg)</Label>
            <Input
              id="diastolicBP"
              type="number"
              placeholder="e.g., 80"
              value={formData.diastolicBP}
              onChange={(e) => handleInputChange("diastolicBP", e.target.value)}
            />
          </div>

          {/* Cholesterol */}
          <div className="space-y-2">
            <Label htmlFor="cholesterol">Cholesterol (mg/dL)</Label>
            <Input
              id="cholesterol"
              type="number"
              placeholder="e.g., 200"
              value={formData.cholesterol}
              onChange={(e) => handleInputChange("cholesterol", e.target.value)}
            />
          </div>

          {/* BMI */}
          <div className="space-y-2">
            <Label htmlFor="bmi">BMI</Label>
            <Input
              id="bmi"
              type="number"
              step="0.1"
              placeholder="e.g., 24.5"
              value={formData.bmi}
              onChange={(e) => handleInputChange("bmi", e.target.value)}
            />
          </div>

          {/* Email */}
          <div className="space-y-2 md:col-span-2">
            <Label htmlFor="email">Email *</Label>
            <Input
              id="email"
              type="email"
              placeholder="your.email@example.com"
              value={formData.email}
              onChange={(e) => handleInputChange("email", e.target.value)}
            />
          </div>
        </div>

        {/* Checkboxes */}
        <div className="flex gap-6">
          <div className="flex items-center space-x-2">
            <Checkbox
              id="smoker"
              checked={formData.smoker}
              onCheckedChange={(checked) =>
                handleInputChange("smoker", checked)
              }
            />
            <Label htmlFor="smoker" className="cursor-pointer">
              Smoker
            </Label>
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox
              id="diabetes"
              checked={formData.diabetes}
              onCheckedChange={(checked) =>
                handleInputChange("diabetes", checked)
              }
            />
            <Label htmlFor="diabetes" className="cursor-pointer">
              Diabetes
            </Label>
          </div>
        </div>

        {/* Location */}
        <div className="space-y-2">
          <Label>Location</Label>
          <Button
            type="button"
            variant="outline"
            onClick={handleLocationClick}
            className="w-full"
          >
            <MapPin className="w-4 h-4 mr-2" />
            {formData.latitude && formData.longitude
              ? `Location: ${formData.latitude}, ${formData.longitude}`
              : "Get Current Location"}
          </Button>

          {/* NEW: editable inputs that show latitude & longitude */}
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div className="space-y-2">
              <Label htmlFor="latitude">Latitude</Label>
              <Input
                id="latitude"
                placeholder="Latitude"
                value={formData.latitude}
                onChange={(e) => handleInputChange("latitude", e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="longitude">Longitude</Label>
              <Input
                id="longitude"
                placeholder="Longitude"
                value={formData.longitude}
                onChange={(e) => handleInputChange("longitude", e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Submit */}
        <Button
          type="submit"
          className="w-full bg-gradient-primary hover:opacity-90 text-white py-6 text-lg"
        >
          Analyze My Health
        </Button>
      </form>
    </Card>
  );
};

export default HealthForm;
