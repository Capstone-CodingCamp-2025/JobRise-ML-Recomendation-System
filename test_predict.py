import requests
import json

# Define the base URL of your FastAPI app
BASE_URL = "https://jobrise-ml-recomendation-system-production-9992.up.railway.app/predict_by_skills"  # Change if running on a different host

# Define the test cases (including non-tech related roles)
test_cases = [
    {
        "skills": ["Python", "Data Analysis", "SQL"],
        "top_k": 15
    },
    {
        "skills": ["Java", "Spring Boot", "Microservices", "Docker"],
        "top_k": 15
    },
    {
        "skills": ["Machine Learning", "Deep Learning", "TensorFlow", "Python"],
        "top_k": 15
    },
    {
        "skills": ["Big Data", "Hadoop", "Spark", "Data Science", "Python"],
        "top_k": 15
    },
    {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
        "top_k": 15
    },
    {
        "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD"],
        "top_k": 15
    },
    {
        "skills": ["Project Management", "Leadership", "Communication", "Agile"],
        "top_k": 15
    },
    {
        "skills": ["PostgreSQL", "MySQL", "SQL", "API Development", "Backend Development"],
        "top_k": 15
    },
    {
        "skills": ["UX Design", "UI Design", "Figma", "Adobe XD", "Wireframing"],
        "top_k": 15
    },
    {
        "skills": ["SEO", "Google Analytics", "Digital Marketing", "Content Marketing"],
        "top_k": 15
    },
    {
        "skills": ["Cybersecurity", "Penetration Testing", "Network Security", "Ethical Hacking"],
        "top_k": 15
    },
    {
        "skills": ["Product Management", "Market Research", "Business Strategy", "Agile", "Scrum"],
        "top_k": 15
    },
    {
        "skills": ["ETL", "Data Pipeline", "Apache Kafka", "Airflow", "Python"],
        "top_k": 15
    },
    {
        "skills": ["HTML", "CSS", "JavaScript", "Node.js", "Express", "MongoDB"],
        "top_k": 15
    },
    {
        "skills": ["Unity", "C#", "Game Development", "Unreal Engine", "3D Modeling"],
        "top_k": 15
    },
    # Non-Tech Related Test Cases
    {
        "skills": ["Leadership", "Team Management", "Employee Engagement", "Human Resources"],
        "top_k": 15
    },
    {
        "skills": ["Marketing Strategy", "Content Creation", "Social Media", "Brand Management"],
        "top_k": 15
    },
    {
        "skills": ["Sales Management", "Negotiation", "Client Relationship", "Salesforce"],
        "top_k": 15
    },
    {
        "skills": ["Customer Service", "Problem Solving", "Communication", "Conflict Resolution"],
        "top_k": 15
    },
    {
        "skills": ["Business Development", "Market Research", "B2B Sales", "Lead Generation"],
        "top_k": 15
    },
    {
        "skills": ["Financial Analysis", "Excel", "Budgeting", "Risk Management"],
        "top_k": 15
    },
    {
        "skills": ["Supply Chain Management", "Logistics", "Inventory Control", "Procurement"],
        "top_k": 15
    },
    {
        "skills": ["Public Relations", "Media Relations", "Event Planning", "Crisis Management"],
        "top_k": 15
    },
    {
        "skills": ["Operations Management", "Process Optimization", "Lean Management", "Cost Reduction"],
        "top_k": 15
    },
    {
        "skills": ["Legal Research", "Contract Negotiation", "Corporate Law", "Compliance"],
        "top_k": 15
    }
]

# Function to send the request and get the response
def fetch_job_titles(skills, top_k):
    response = requests.post(BASE_URL, json={"skills": skills, "top_k": top_k})

    if response.status_code == 200:
        data = response.json()
        if "recommendations" in data:
            job_titles = [job["title"] for job in data["recommendations"]]
            return job_titles
        else:
            return ["No recommendations found."]
    else:
        return ["Error: Unable to fetch data."]

# Loop through the test cases and print job titles

with open("job_recommendations2.txt", "w") as f:
    # Loop through the test cases and save job titles to the file
    for i, case in enumerate(test_cases, 1):
        f.write(f"Test Case {i} (Skills: {', '.join(case['skills'])}):\n")
        job_titles = fetch_job_titles(case["skills"], case["top_k"])

        f.write("\nTop Job Titles:\n")
        if job_titles:
            for idx, title in enumerate(job_titles, 1):
                f.write(f"{idx}. {title}\n")
        else:
            f.write("No job titles found.\n")

        f.write("\n" + "="*50 + "\n")

print("Job recommendations have been saved to 'job_recommendations2.txt'.")
