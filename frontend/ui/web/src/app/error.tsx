"use client";

import Lottie from "@/components/ui/lottie";
import lottieData from "public/AnimationError.json";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import { siteConfig } from "@/config/site";

export default function Error({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  const emailSubject = encodeURIComponent("Client-Side Error Report");
  const sendBody =
    `Dear Support Team,\n\n` +
    `An error occurred in the application. Below are the details:\n\n` +
    `Error Message: ${error.message}\n` +
    `Stack Trace: ${error.stack}\n\n` +
    `Additional Information: \n` +
    `- Browser: ${typeof navigator !== "undefined" ? navigator.userAgent : "N/A"}\n` +
    `- Time: ${new Date().toISOString()}\n\n` +
    `Please let me know if further information is required.\n\n` +
    `Best regards,\n[Your Name]`;
  const [emailBody, setEmailBody] = useState(sendBody);

  const mailtoLink = `mailto:support@robustai.dev?subject=${emailSubject}&body=${encodeURIComponent(emailBody)}`;

  return (
    <div className="w-full flex justify-center items-center">
      <div className="shadow-lg rounded-lg p-6 max-w-2xl flex flex-col items-center">
        <h1 className="text-2xl font-bold text-red-600 mb-4">
          An Error Occurred That Cannot Be Solved
        </h1>
        <Lottie
          className="w-48 mb-6"
          animationData={lottieData}
          options={{ loop: false, autoplay: true }}
        />
        <p className="text-lg text-center text-red-600 leading-relaxed mb-4">
          An error occurred. Please report this issue to our{" "}
          <a
            href={siteConfig.links.github + "/issues"}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-blue-400"
          >
            GitHub repo
          </a>{" "}
          or email our technical support using the button below.
        </p>
        <p className="text-gray-500 text-sm mb-6">
          <strong>Error:</strong> {error.message}
        </p>
        <div className="w-full mb-4">
          <Label
            htmlFor="message"
            className="text-sm font-medium text-gray-700"
          >
            Your Message
          </Label>
          <Textarea
            onChange={(e) => setEmailBody(e.target.value)}
            className="mt-2 h-40 resize-none"
            value={emailBody}
            placeholder="Type your message here."
            id="message"
          />
          <p className="text-xs text-gray-500 mt-1">
            Your message will be copied to the support team.
          </p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => reset()}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
          <a
            href={mailtoLink}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Email Support
          </a>
        </div>
      </div>
    </div>
  );
}
