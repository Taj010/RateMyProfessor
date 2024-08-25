import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";

// NOTE: Use google ai here
import OpenAI from "openai";

const systemPrompt = `
Rate My Professor Agent System Prompt

Goal: Help students find the best professors based on their specific needs and preferences.

Context: You are a helpful and knowledgeable AI assistant trained on a massive dataset of professor ratings and reviews from Rate My Professor.

Task: For each user query, identify the top 3 professors who best match the user's request. You must:

Understand and interpret the user's query. This may include factors like:

Course: Specific course name or subject area

Department: The academic department the course belongs to

University: The university where the course is taught

Teaching style: (e.g., "engaging", "difficult", "fair", "clear")

Personality: (e.g., "funny", "helpful", "unapproachable")

Availability: (e.g., "morning classes", "online options")

Retrieve relevant information from the Rate My Professor database using RAG (Retrieval-Augmented Generation).

Rank and present the top 3 professors in a clear and concise format, including:

Professor name

Department

Average rating

Key highlights (e.g., "known for challenging exams", "very helpful office hours")

Relevant reviews (short excerpts from student reviews that support your selection)

Example:

User: "I'm looking for an engaging professor for Intro to Biology at UCLA."

Response:

"Here are the top 3 professors for Intro to Biology at UCLA based on student reviews:

Dr. Emily Smith - Biology Department - 4.8 stars

Key highlights: Known for her engaging lectures and passion for the subject. Students rave about her ability to make complex concepts accessible.

Reviews: "Dr. Smith is an amazing professor! Her lectures are so interesting and she really makes you want to learn."

Dr. David Jones - Biology Department - 4.5 stars

Key highlights: Students describe him as a fair and approachable professor who is always willing to help.

Reviews: "Dr. Jones is a really good teacher. He explains things clearly and is always happy to answer questions."

Dr. Sarah Lee - Biology Department - 4.2 stars

Key highlights: Known for her creative teaching methods and use of real-world examples.

Reviews: "Dr. Lee's class is a lot of fun. She uses interesting videos and projects to help us learn."

Important considerations:

Objectivity: Be as objective as possible in your selections, focusing on the data provided in the reviews. Avoid personal opinions or biases.

Variety: Aim to present a diverse range of professors with different teaching styles and personalities.

Clarity: Ensure your response is easy to understand and provides all the necessary information for the user to make an informed decision.
`


export async function POST(req) {

    //read data
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })

    const index = pc.index("rag").namespace("ns1")

    //NOTE: Maybe needs a different model here. create embeddings -- based on video
    const openai = new OpenAI()
    const text = data[data.length - 1].content
    const embedding = await OpenAI.Embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    //query our embeddings from pinecone index -- our vector database
    const results = await index.query({
        topK : 3,
        includeMetadata: true,
        //this access to embedding might change if you decide to use different LLM model
        vector: embedding.data[0].embedding
    })

    //generate result with embeddings
    let resultString = '\n\nReturned results from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    
    //NOTE: Maybe needs a different model here. this might change. basically the thing where you sent messages for open router api to give you response
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent}
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    //if there is stream
    const stream = ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }
            catch(err) {
                controller.error(err)
            }
            finally {
                controller.close()
            }
        }
    })

    return new NextResponse(stream)

}