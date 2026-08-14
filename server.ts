import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import { generateDynamicProposalForBrief } from './src/utils/dynamicProposalGenerator';
import { resolveVisualsForProduct } from './src/utils/categoryImageResolver';

dotenv.config();

function getGenAIClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn('GEMINI_API_KEY is not set in environment.');
  }
  return new GoogleGenAI({
    apiKey: apiKey || '',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // JSON payload parser with generous limit for audio / base64
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
  });

  // Endpoint 1: Extract Audio Requirements
  app.post('/api/extract-audio', async (req, res) => {
    try {
      const { base64Data, mimeType, transcriptText, category, topic } = req.body;
      const ai = getGenAIClient();

      const systemPrompt = `You are a Chief Furniture Design Director & Trend Strategist.
Your role is to analyze voice memo/audio recordings or design meeting transcripts submitted by furniture product planners.
Do not simply summarize the voice recording. Extract the exact design and strategic parameters.

Extract the following items strictly. If any item is NOT mentioned or cannot be verified from the input, mark it strictly as "미지정" (Unspecified in Korean).

1. product (제품): Furniture category/type
2. goal (목표): Project objective / planning goal
3. target (타깃): Target customer demographic / lifestyle
4. market (시장): Target market region (e.g. Global, Korea, Europe, North America)
5. year (연도): Target launch year (e.g. 2027)
6. colorReq (컬러 요구): Specific color palette or tone mentioned
7. materialReq (소재 요구): Specific materials, finishes, or textures
8. formReq (형태 요구): Specific silhouettes, proportions, curves/lines, or ergonomic posture
9. refBrands (참고 브랜드): Mentioned furniture or luxury design brands
10. refExhibitions (참고 전시): Mentioned design fairs or exhibitions (e.g. Salone del Mobile, Maison&Objet)
11. mandatoryReq (반드시 반영할 요구사항): Essential constraints or must-have features

Output valid JSON matching this schema:
{
  "product": string,
  "goal": string,
  "target": string,
  "market": string,
  "year": string,
  "colorReq": string,
  "materialReq": string,
  "formReq": string,
  "refBrands": string,
  "refExhibitions": string,
  "mandatoryReq": string
}`;

      const contents: any[] = [];
      if (base64Data && mimeType) {
        contents.push({
          inlineData: {
            data: base64Data,
            mimeType: mimeType || 'audio/mp3',
          },
        });
      }

      let textPrompt = `Context:\nCategory: ${category || '미지정'}\nTopic: ${topic || '미지정'}\n\n`;
      if (transcriptText) {
        textPrompt += `Transcript / Notes:\n${transcriptText}\n\n`;
      }
      textPrompt += `Please analyze the input and extract all 11 furniture planning requirements in Korean. If any item was not specified, output "미지정".`;

      contents.push({ text: textPrompt });

      const response = await ai.models.generateContent({
        model: 'gemini-3.7-flash',
        contents: contents,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              product: { type: Type.STRING },
              goal: { type: Type.STRING },
              target: { type: Type.STRING },
              market: { type: Type.STRING },
              year: { type: Type.STRING },
              colorReq: { type: Type.STRING },
              materialReq: { type: Type.STRING },
              formReq: { type: Type.STRING },
              refBrands: { type: Type.STRING },
              refExhibitions: { type: Type.STRING },
              mandatoryReq: { type: Type.STRING },
            },
            required: [
              'product',
              'goal',
              'target',
              'market',
              'year',
              'colorReq',
              'materialReq',
              'formReq',
              'refBrands',
              'refExhibitions',
              'mandatoryReq',
            ],
          },
        },
      });

      const extracted = JSON.parse(response.text?.trim() || '{}');
      res.json({ success: true, extracted });
    } catch (error: any) {
      console.error('Audio extraction error:', error);
      res.status(500).json({
        success: false,
        error: error.message || 'Failed to extract audio requirements',
      });
    }
  });

  // Endpoint 2: Generate / Synthesize Project Brief
  app.post('/api/generate-brief', async (req, res) => {
    try {
      const {
        productCategory,
        customCategory,
        projectTopic,
        targetMarket,
        targetCustomer,
        customCustomer,
        targetYear,
        extractedAudio,
      } = req.body;

      const ai = getGenAIClient();

      const systemPrompt = `You are a Senior Furniture Design Strategist.
Synthesize the user's project input and extracted audio requirements into a crisp, authoritative PROJECT BRIEF in Korean.

Format:
- product: Final refined product type (e.g. "Lounge Chair (프리미엄 거실용 라운지체어)")
- projectGoal: Clear 1-2 sentence project objective for new product development
- target: Target user segment with lifestyle characterization
- market: Target market region
- year: Target launch year
- keyRequirements: Array of 3 to 5 clear, actionable requirements combining user inputs, audio notes, ergonomic/functional and aesthetic needs
- references: Array of 3 to 5 reputable furniture brands, designers, or design exhibitions (e.g. Salone del Mobile, Cassina, B&B Italia, Minotti, Maison&Objet) relevant to the brief.`;

      const prompt = `Input Parameters:
- Product Category: ${customCategory || productCategory}
- Project Topic: ${projectTopic}
- Target Market: ${targetMarket}
- Target Customer: ${customCustomer || targetCustomer}
- Target Year: ${targetYear}
- Extracted Audio Details: ${JSON.stringify(extractedAudio || {})}

Synthesize into an authoritative Project Brief JSON.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.7-flash',
        contents: prompt,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              product: { type: Type.STRING },
              projectGoal: { type: Type.STRING },
              target: { type: Type.STRING },
              market: { type: Type.STRING },
              year: { type: Type.STRING },
              keyRequirements: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              references: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
            },
            required: [
              'product',
              'projectGoal',
              'target',
              'market',
              'year',
              'keyRequirements',
              'references',
            ],
          },
        },
      });

      const brief = JSON.parse(response.text?.trim() || '{}');
      res.json({ success: true, brief });
    } catch (error: any) {
      console.error('Brief generation error (using tailored dynamic brief):', error);
      const product = req.body.customCategory || req.body.productCategory || 'Living Furniture';
      const target = req.body.customCustomer || req.body.targetCustomer || 'Premium';
      const market = req.body.targetMarket || 'Global';
      const year = req.body.targetYear || '2027';
      const topic = req.body.projectTopic || `${year}년 글로벌 프리미엄 리빙 공간을 위한 ${product} 디자인 전략 및 CMF 차별화 수립`;

      const tailoredBrief = {
        product: `${product} (신규 개발 라인업)`,
        projectGoal: topic,
        target: `${target} Lifestyle (심미안과 기능적 만족을 추구하는 사용자)`,
        market: `${market} (밀라노/파리/서울/뉴욕 프리미엄 시장)`,
        year,
        keyRequirements: [
          `${product}의 조형적 존재감과 인체공학적/공간적 편의성의 균형`,
          '2027 글로벌 리빙 트렌드를 반영한 차별화된 CMF 및 친환경 마감재 융합',
          '360도 어디서나 완성도 높은 실루엣과 지속 가능한 내구성 확보',
        ],
        references: ['Salone del Mobile.Milano 2025–2026', 'B&B Italia', 'Cassina', 'Minotti', 'Poliform'],
      };

      res.json({ success: true, brief: tailoredBrief });
    }
  });

  // Endpoint 3: Full Furniture Trend Research and Product Proposal Generation
  app.post('/api/generate-proposal', async (req, res) => {
    const { brief, customGuidance } = req.body || {};
    try {
      if (!brief) {
        return res.status(400).json({ success: false, error: 'Project brief is required' });
      }

      const ai = getGenAIClient();

      const systemPrompt = `You are a Global Living & Furniture Trend Research Director and Design Strategist.
Generate a complete, professional, high-credibility Furniture Trend & Product Planning Proposal (리빙가구 트렌드 기반 상품기획 제안서).

The tone MUST be professional, authoritative, analytical, and design-directed (as written by a Milan/Paris/Seoul Furniture Design Director).
NEVER use empty buzzwords ("혁신적", "획기적", "미래지향적", "특별한", "트렌디한", "세련된").
Always write in the logical flow: What changed -> Why it matters -> How to apply in furniture design.

STRICT RULES & CONSTRAINTS:
1. OVERVIEW: Maximum 3 sentences summarizing the strategic intent of this product development.
2. 5 KEY TRENDS: Exactly 5 non-overlapping, high-impact living trends.
   - For each trend, provide:
     * name: Distinctive Trend Name in Korean with English translation
     * whatIsChanging: What is changing in living lifestyles / furniture forms
     * whyItMatters: Why this shift is strategically critical
     * furnitureImplication: Direct design impact on structure, silhouette, or mechanics
     * evidence: Real-world observation evidence (from Salone del Mobile, Milan Design Week, Maison&Objet, or top design brands)
     * evaluation: { recency: 'HIGH'|'MEDIUM'|'LOW', repetition: 'HIGH'|'MEDIUM'|'LOW', spread: 'HIGH'|'MEDIUM'|'LOW', relevance: 'HIGH'|'MEDIUM'|'LOW', differentiation: 'HIGH'|'MEDIUM'|'LOW' }
3. CMF + FORM DIRECTION:
   - colors: 5 to 7 curated trend colors with 'name', 'hex' (valid 6-digit hex code e.g. #C8A97E), and 'application'
   - materials: 4 to 6 core materials with 'material', 'character', 'recommendedApplication', 'whyNow'
   - forms: 4 to 6 form keywords (e.g. Monolithic Curved Volume, Low-Profile Cradle, Floating Horizon, Seamless Interlock) with 1-sentence 'implication'
4. BRAND & EXHIBITION SIGNALS:
   - 3 to 5 verified case studies from world-class design houses (e.g. B&B Italia, Cassina, Minotti, Poltrona Frau, Vitra, Moroso, Flexform, Edra, Ligne Roset, Kvadrat, Formafantasma) or major fairs (Salone del Mobile.Milano, Maison&Objet).
   - Set sourceReliability: 'LEVEL A' for official brand/fair archive or 'LEVEL B' for top architectural/design publications (Wallpaper*, Dezeen, Interni).
   - Provide whyItMatters and relatedTrend.
5. FROM TREND TO PRODUCT OPPORTUNITY:
   - Exactly 3 Product Opportunities.
   - Structure each: Observed Change -> Design Insight -> User Value -> Furniture Opportunity.
6. 3 DESIGN DIRECTIONS (3 distinct strategic concepts):
   - CONCEPT 01, CONCEPT 02, CONCEPT 03.
   - Each concept must have a clear differentiated strategy (differing in form, material, structure, or user interaction).
   - Provide: name, conceptStatement, designKeywords (3-5), color, material, form, targetExperience, differentiation, relatedTrend, imagePrompt (detailed photorealistic prompt for editorial visualization).
7. CONCEPT COMPARISON & RECOMMENDATION:
   - Comparison matrix with 6 rows:
     * Design Character (디자인 캐릭터)
     * Target User (주요 타깃 적합성)
     * Trend Relevance (트렌드 부합도)
     * Differentiation (차별화 포인트)
     * Product Feasibility (양산 및 생산성)
     * Market Potential (글로벌 시장성)
   - Recommendation: 1 recommended concept with exactly 3 clear rationale bullet points.
8. SPACE APPLICATION:
   - Application of the recommended concept in a high-end space (e.g. Luxury Living Room, Penthouse, Boutique Hotel Lounge) with spaceType, description, and imagePrompt.
9. FINAL PRODUCT PROPOSAL ("WHAT SHOULD WE DESIGN?"):
   - whatShouldWeDesign: 1 powerful sentence declaring the final product development direction.
   - product, target, designLanguage, color, material, form, userExperience, differentiation, whyNow.
   - nextActions: Exactly 3 immediate next steps for product engineering, prototype mock-up, and fair launch.
10. DESIGN RESPONSE LOGIC:
   - userRequirement (User's core desire) -> trendInsight (Discovered design trend) -> designResponse (Definitive form/material resolution).

Output strict JSON.`;

      const prompt = `Project Brief for Analysis:
Product: ${brief.product}
Project Goal: ${brief.projectGoal}
Target Customer: ${brief.target}
Market: ${brief.market}
Target Year: ${brief.year}
Key Requirements: ${JSON.stringify(brief.keyRequirements)}
References: ${JSON.stringify(brief.references)}
${customGuidance ? `Custom Directives: ${customGuidance}` : ''}

Generate the complete Furniture Trend & Product Proposal JSON in Korean with international design terminology.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.7-flash',
        contents: prompt,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              overview: { type: Type.STRING },
              trends: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    number: { type: Type.STRING },
                    name: { type: Type.STRING },
                    whatIsChanging: { type: Type.STRING },
                    whyItMatters: { type: Type.STRING },
                    furnitureImplication: { type: Type.STRING },
                    evidence: { type: Type.STRING },
                    evaluation: {
                      type: Type.OBJECT,
                      properties: {
                        recency: { type: Type.STRING },
                        repetition: { type: Type.STRING },
                        spread: { type: Type.STRING },
                        relevance: { type: Type.STRING },
                        differentiation: { type: Type.STRING },
                      },
                      required: [
                        'recency',
                        'repetition',
                        'spread',
                        'relevance',
                        'differentiation',
                      ],
                    },
                  },
                  required: [
                    'id',
                    'number',
                    'name',
                    'whatIsChanging',
                    'whyItMatters',
                    'furnitureImplication',
                    'evidence',
                    'evaluation',
                  ],
                },
              },
              cmf: {
                type: Type.OBJECT,
                properties: {
                  colors: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        hex: { type: Type.STRING },
                        application: { type: Type.STRING },
                      },
                      required: ['name', 'hex', 'application'],
                    },
                  },
                  materials: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        material: { type: Type.STRING },
                        character: { type: Type.STRING },
                        recommendedApplication: { type: Type.STRING },
                        whyNow: { type: Type.STRING },
                      },
                      required: [
                        'material',
                        'character',
                        'recommendedApplication',
                        'whyNow',
                      ],
                    },
                  },
                  forms: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        keyword: { type: Type.STRING },
                        implication: { type: Type.STRING },
                      },
                      required: ['keyword', 'implication'],
                    },
                  },
                },
                required: ['colors', 'materials', 'forms'],
              },
              brandSignals: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    brandOrExhibition: { type: Type.STRING },
                    productOrProject: { type: Type.STRING },
                    year: { type: Type.STRING },
                    source: { type: Type.STRING },
                    sourceReliability: { type: Type.STRING },
                    whyItMatters: { type: Type.STRING },
                    relatedTrend: { type: Type.STRING },
                  },
                  required: [
                    'brandOrExhibition',
                    'productOrProject',
                    'year',
                    'source',
                    'sourceReliability',
                    'whyItMatters',
                    'relatedTrend',
                  ],
                },
              },
              opportunities: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    number: { type: Type.STRING },
                    title: { type: Type.STRING },
                    observedChange: { type: Type.STRING },
                    designInsight: { type: Type.STRING },
                    userValue: { type: Type.STRING },
                    furnitureOpportunity: { type: Type.STRING },
                  },
                  required: [
                    'id',
                    'number',
                    'title',
                    'observedChange',
                    'designInsight',
                    'userValue',
                    'furnitureOpportunity',
                  ],
                },
              },
              concepts: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    conceptNumber: { type: Type.STRING },
                    name: { type: Type.STRING },
                    conceptStatement: { type: Type.STRING },
                    designKeywords: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                    },
                    color: { type: Type.STRING },
                    material: { type: Type.STRING },
                    form: { type: Type.STRING },
                    targetExperience: { type: Type.STRING },
                    differentiation: { type: Type.STRING },
                    relatedTrend: { type: Type.STRING },
                    imagePrompt: { type: Type.STRING },
                  },
                  required: [
                    'id',
                    'conceptNumber',
                    'name',
                    'conceptStatement',
                    'designKeywords',
                    'color',
                    'material',
                    'form',
                    'targetExperience',
                    'differentiation',
                    'relatedTrend',
                    'imagePrompt',
                  ],
                },
              },
              comparison: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    criteria: { type: Type.STRING },
                    concept01: { type: Type.STRING },
                    concept02: { type: Type.STRING },
                    concept03: { type: Type.STRING },
                  },
                  required: ['criteria', 'concept01', 'concept02', 'concept03'],
                },
              },
              recommendation: {
                type: Type.OBJECT,
                properties: {
                  conceptId: { type: Type.STRING },
                  conceptNumber: { type: Type.STRING },
                  conceptName: { type: Type.STRING },
                  reasons: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: [
                  'conceptId',
                  'conceptNumber',
                  'conceptName',
                  'reasons',
                ],
              },
              spaceConcept: {
                type: Type.OBJECT,
                properties: {
                  spaceType: { type: Type.STRING },
                  description: { type: Type.STRING },
                  imagePrompt: { type: Type.STRING },
                },
                required: ['spaceType', 'description', 'imagePrompt'],
              },
              finalProposal: {
                type: Type.OBJECT,
                properties: {
                  whatShouldWeDesign: { type: Type.STRING },
                  product: { type: Type.STRING },
                  target: { type: Type.STRING },
                  designLanguage: { type: Type.STRING },
                  color: { type: Type.STRING },
                  material: { type: Type.STRING },
                  form: { type: Type.STRING },
                  userExperience: { type: Type.STRING },
                  differentiation: { type: Type.STRING },
                  whyNow: { type: Type.STRING },
                  nextActions: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: [
                  'whatShouldWeDesign',
                  'product',
                  'target',
                  'designLanguage',
                  'color',
                  'material',
                  'form',
                  'userExperience',
                  'differentiation',
                  'whyNow',
                  'nextActions',
                ],
              },
              responseLogic: {
                type: Type.OBJECT,
                properties: {
                  userRequirement: { type: Type.STRING },
                  trendInsight: { type: Type.STRING },
                  designResponse: { type: Type.STRING },
                },
                required: [
                  'userRequirement',
                  'trendInsight',
                  'designResponse',
                ],
              },
            },
            required: [
              'overview',
              'trends',
              'cmf',
              'brandSignals',
              'opportunities',
              'concepts',
              'comparison',
              'recommendation',
              'spaceConcept',
              'finalProposal',
              'responseLogic',
            ],
          },
        },
      });

      const parsed = JSON.parse(response.text?.trim() || '{}');

      // Add high quality curated visuals/imagery for concepts and brand signals tailored to the product category
      const visuals = resolveVisualsForProduct(brief.product, brief.projectGoal);

      if (Array.isArray(parsed.concepts)) {
        parsed.concepts = parsed.concepts.map((c: any, index: number) => ({
          ...c,
          imageUrl:
            c.imageUrl && c.imageUrl.startsWith('http')
              ? c.imageUrl
              : visuals.conceptImages[index % visuals.conceptImages.length],
        }));
      }

      if (Array.isArray(parsed.brandSignals)) {
        parsed.brandSignals = parsed.brandSignals.map((b: any, index: number) => ({
          ...b,
          imageUrl:
            b.imageUrl && b.imageUrl.startsWith('http')
              ? b.imageUrl
              : visuals.brandSignalImages[index % visuals.brandSignalImages.length],
        }));
      }

      if (parsed.spaceConcept) {
        parsed.spaceConcept.imageUrl =
          parsed.spaceConcept.imageUrl && parsed.spaceConcept.imageUrl.startsWith('http')
            ? parsed.spaceConcept.imageUrl
            : visuals.spaceImage;
      }

      const fullProposal = {
        id: `proposal-${Date.now()}`,
        createdAt: new Date().toISOString().split('T')[0],
        projectBrief: brief,
        ...parsed,
      };

      res.json({ success: true, proposal: fullProposal });
    } catch (error: any) {
      console.error('Proposal generation error (falling back to tailored dynamic synthesis):', error);
      try {
        const dynamicProposal = generateDynamicProposalForBrief(brief, customGuidance);
        res.json({ success: true, proposal: dynamicProposal });
      } catch (synthError: any) {
        res.status(500).json({
          success: false,
          error: synthError.message || 'Failed to generate trend proposal',
        });
      }
    }
  });

  // Vite integration
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`LIVING TREND PROPOSAL Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
