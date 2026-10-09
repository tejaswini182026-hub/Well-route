import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialize Gemini client
let ai: GoogleGenAI | null = null;
function getGeminiClient() {
  if (!ai) {
    const key = process.env.GEMINI_API_KEY;
    // Note: Do not throw an error during startup, but only when used.
    if (!key) {
      throw new Error("GEMINI_API_KEY environment variable is required to power the AI Support Navigator.");
    }
    ai = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return ai;
}

// API endpoint for WellRoute Navigator chat
app.post("/api/navigator", async (req, res) => {
  try {
    const { message, history, context } = req.body || {};

    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Message is required." });
    }

    const safeContext = context || {};

    // Compile dynamic context from state and localStorage properties
    let contextText = "USER PLATFORM PROGRESS & PROFILE CONTEXT:\n";
    if (safeContext.userProfile) {
      contextText += `- Real Name/Nickname: ${safeContext.userProfile.name || "Anonymous"}\n`;
      contextText += `- Onboarding Completed Status: ${safeContext.userProfile.completedOnboarding ? "Yes" : "No"}\n`;
      contextText += `- Age Group: ${safeContext.userProfile.age || "Not defined"}\n`;
      contextText += `- Occupation Roles: ${safeContext.userProfile.occupation || "Not defined"}\n`;
      contextText += `- Interactive Support Preferences: Style: ${safeContext.userProfile.supportStyle || "Not defined"}, Comm: ${safeContext.userProfile.commStyle || "Not defined"}\n`;
      contextText += `- Self-identified Well-being Goals: ${JSON.stringify(safeContext.userProfile.goals || [])}\n`;
    } else if (safeContext.userName) {
      contextText += `- Name: ${safeContext.userName || "Anonymous"}\n`;
    }

    if (safeContext.currentPathway) {
      contextText += `\nCURRENT CALIBRATED PATIENT-CENTRIC PATHWAY:\n`;
      contextText += `- Active Road: ${safeContext.currentPathway.pathway || "No active pathway"}\n`;
      contextText += `- Latest Screener Score Index: ${safeContext.currentPathway.score || "No score yet"}\n`;
      contextText += `- Calibration Confidence: ${safeContext.currentPathway.confidence || 0}%\n`;
      contextText += `- Calibrated Pathway Steps/Directives: ${JSON.stringify(safeContext.currentPathway.steps || [])}\n`;
      contextText += `- Clinical Explanation: ${safeContext.currentPathway.explanation || ""}\n`;
    } else if (safeContext.currentPath) {
      contextText += `\nCURRENT WELLNESS PATHWAY TITLE: ${safeContext.currentPath}\n`;
    }

    if (Array.isArray(safeContext.assessmentHistory) && safeContext.assessmentHistory.length > 0) {
      contextText += `\nHISTORICAL SCREENER ASSESSMENTS TAKE RECORDS:\n`;
      safeContext.assessmentHistory.forEach((item: any, i: number) => {
        contextText += `  - Entry #${i+1}: Date: ${item.timestamp || "recent"}, Distress Score: ${item.score ?? "N/A"}, Resulting Path: ${item.pathway || "N/A"}\n`;
      });
    } else {
      contextText += `\nASSESSMENT RECORDS: No formal diagnostic screening has been recorded yet.\n`;
    }

    if (Array.isArray(safeContext.savedResources) && safeContext.savedResources.length > 0) {
      contextText += `\nBOOKMARKED WELLNESS RESOURCES (LOCAL DATABASE FAVS):\n`;
      safeContext.savedResources.forEach((item: any) => {
        contextText += `  - ${item.name || item.resourceId || "Coping Guide"}\n`;
      });
    } else {
      contextText += `\nBOOKMARKED RESOURCES: No coping resources saved yet.\n`;
    }

    if (Array.isArray(safeContext.userProgress) && safeContext.userProgress.length > 0) {
      contextText += `\nSOMATIC FEEDBACK LOGS & HABIT PROGRESSION:\n`;
      safeContext.userProgress.slice(0, 10).forEach((item: any) => {
        contextText += `  - Status sentiment rating: ${item.sentimentRating || "Logged"} / Notes: ${item.notes || "None"}\n`;
      });
    }

    if (Array.isArray(safeContext.logs) && safeContext.logs.length > 0) {
      contextText += `\nINSTANT CHAT-IN STATE MOOD PULSES:\n`;
      safeContext.logs.slice(0, 10).forEach((log: any) => {
        contextText += `  - [${log.timestamp || "today"}]: Feel ${log.mood || "recorded"} ${log.emoji || ""}\n`;
      });
    }

    const systemInstruction = 
      "You are the Almost Okay Navigator, an empathetic, candid, and grounded mental health navigation companion in the app 'Almost okay'. " +
      "Your vibe is authentic, validating, and conversational (Gen Z friendly, warm, zero toxic positivity, no corporate wellness jargon, zero clinical judgment). " +
      "You help users understand their feelings, navigate coping pathways, suggest grounding techniques, and gently direct to resources or crisis channels when appropriate. " +
      "Use the user's stored information to make them feel heard and validated. Avoid diagnosing medical conditions or claiming to replace human therapy.\n\n" +
      "Below is the live real-time sandbox context compiled from the user's secure browser session. Adapt your vocabulary to reflect these facts directly:\n" +
      contextText;

    // Build the messages list for Chat API
    const contents: any[] = [];
    if (Array.isArray(history) && history.length > 0) {
      history.forEach((h: any) => {
        if (h && typeof h.message === "string" && h.message.trim()) {
          contents.push({
            role: h.role === "user" ? "user" : "model",
            parts: [{ text: h.message.trim() }],
          });
        }
      });
    }

    // Ensure the message history doesn't start with a 'model' turn
    while (contents.length > 0 && contents[0].role !== "user") {
      contents.shift();
    }

    contents.push({
      role: "user",
      parts: [{ text: message.trim() }],
    });

    const candidateModels = [
      "gemini-3.5-flash-lite",
      "gemini-3-flash-preview",
      "gemini-3.8-flash",
      "gemini-flash-latest"
    ];

    let outputText = "";
    let lastGenError: any = null;

    try {
      const client = getGeminiClient();
      for (const modelName of candidateModels) {
        try {
          const response = await client.models.generateContent({
            model: modelName,
            contents: contents,
            config: {
              systemInstruction: systemInstruction,
              temperature: 0.7,
            },
          });
          if (response && response.text && response.text.trim()) {
            outputText = response.text.trim();
            break;
          }
        } catch (mErr: any) {
          lastGenError = mErr;
          console.warn(`Model ${modelName} returned error:`, mErr?.message || mErr);
        }
      }
    } catch (clientErr: any) {
      lastGenError = clientErr;
      console.warn("Client setup or initialization issue:", clientErr?.message || clientErr);
    }

    if (outputText) {
      return res.json({ text: outputText });
    }

    // Fallback response generator if models are unreachable or rate-limited
    const fallbackName = safeContext.userProfile?.name || safeContext.userName || "friend";
    const currentPath = safeContext.currentPathway?.pathway || safeContext.currentPath || "Self-Guided Wellness Pathway";
    
    let fallbackText = `Hello ${fallbackName}. I hear you, and I am right here with you.\n\n`;
    if (message.toLowerCase().includes("breathe") || message.toLowerCase().includes("anxious") || message.toLowerCase().includes("panic")) {
      fallbackText += `When tension runs high, steady grounding can help restore balance. Consider trying the **Box Breathing (4-4-4-4)** exercise available in your Resources tab: inhale for 4 seconds, hold for 4, exhale for 4, and rest for 4.\n\n`;
    } else if (message.toLowerCase().includes("burnout") || message.toLowerCase().includes("tired") || message.toLowerCase().includes("overwhelm")) {
      fallbackText += `Navigating fatigue is tough. You're currently following the **${currentPath}**. Remember that resting is not giving up—it is essential maintenance. What is one small demand you can set aside for the rest of today?\n\n`;
    } else {
      fallbackText += `You are currently aligned with the **${currentPath}**. Every small reflection you make is a meaningful step toward equilibrium.\n\n`;
    }
    fallbackText += `Feel free to explore your curated exercises in the **Resources** tab or retake the pathway assessment whenever your circumstances evolve.`;

    return res.json({ text: fallbackText });
  } catch (error: any) {
    console.error("Gemini API server failure error:", error);
    res.status(200).json({ 
      text: "I am currently adjusting to high traffic, but your progress and pathway remain completely secure. Take a calm breath, and feel free to try your message again in a moment." 
    });
  }
});

// Serve Vite dev server or static distribution files
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Almost okay Backend] running successfully on port ${PORT}`);
  });
}

startServer();
