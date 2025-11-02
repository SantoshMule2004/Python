import numpy as np

# Problem Statement 1 :
# Write a Python program using NumPy to analyze the grades of multiple students across different subjects.

# Your program should:
# Create a list of 5 student names and a list of 3 subjects.
# Generate a 5×3 matrix of random integer grades between 60 and 100 using np.random.randint().
# Display the grades matrix, student names, and subject names.

# Compute and display:
# The average grade for each student (row-wise mean).
# The average grade for each subject (column-wise mean).
# Print each student’s name along with their average grade (rounded to 1 decimal place).
# Determine and print the top-performing student and their average score.

def averageGrade():
    # list of 5 student names and a list of 3 subjects
    students = ["Alice", "Bob", "Charlie", "David", "Eve"]
    subjects = ["Math", "Science", "English"]

    noStud = len(students)
    noSub = len(subjects)

    # 5×3 matrix of random integer grades between 60 and 100
    gradeMatrix = [[0 for _ in range(noSub)] for _ in range(noStud)]
    
    # generating random int bewtween 60 and 100
    for i in range(noStud):
        for j in range(noSub):
            gradeMatrix[i][j] = np.random.randint(60, 100)

    # printing the grade matrix
    for i in range(noStud):
        for j in range(noSub):
            print(gradeMatrix[i][j], end=" ")
        print()

    print()
    # The average grade for each student (row-wise mean).
    topperId , topperAvg = 0, float("-inf")
    for i in range(noStud):
        rowTotal = 0
        for j in range(noSub):
            rowTotal += gradeMatrix[i][j]
        avg = round(rowTotal/noSub, 1)
        if avg > topperAvg:
            topperAvg = avg
            topperId = i
        print(students[i], "average: ", )

    print()

    # The average grade for each subject (column-wise mean).
    for j in range(noSub):
        colTotal = 0
        for i in range(noStud):
            colTotal += gradeMatrix[i][j]
        print(j, " th subject average: ", colTotal/noStud)

    print()
    print("the top-performing student is", students[topperId], "and average is", topperAvg)

# averageGrade()

def averagegradeOfStud():
    # Create student and subject lists
    students = ["Alice", "Bob", "Charlie", "David", "Eve"]
    subjects = ["Math", "Science", "English"]

    # Generate 5×3 matrix of random grades between 60 and 100 (inclusive)
    grades = np.random.randint(60, 101, size=(len(students), len(subjects)))

    # Display grades matrix, student names, and subject names
    print("Subjects:", subjects)
    print("\nGrades Matrix:")
    print(grades)

    # Compute average grade for each student (row-wise mean)
    student_avg = np.mean(grades, axis=1)

    # Compute average grade for each subject (column-wise mean)
    subject_avg = np.mean(grades, axis=0)

    # Display student averages
    print("\nAverage grade for each student:")
    for i in range(len(students)):
        print(f"{students[i]}: {student_avg[i]:.1f}")

    # Display subject averages
    print("\nAverage grade for each subject:")
    for i in range(len(subjects)):
        print(f"{subjects[i]}: {subject_avg[i]:.1f}")

    # Determine and print top-performing student
    top_index = np.argmax(student_avg)
    print(f"\nTop-performing student: {students[top_index]} with an average of {student_avg[top_index]:.1f}")


# Problem Statement 2:
# Write a Python program using NumPy to perform analysis on sales data of different products over six months.

# Your program should:
# Create a list of 6 months (Jan–Jun) and 3 products (Product A, Product B, Product C).
# Generate a 6×3 matrix of random sales figures (between 1000 and 5000) using np.random.randint().
# Display the sales matrix clearly (Months × Products).

# Calculate and display:
# Total monthly sales (sum across products).
# Total sales per product (sum across months).

# Identify and display:
# The best-performing month (highest total sales).
# The best-selling product (highest cumulative sales).

def salesDataAnalysis():
    # Create student and subject lists
    months = ["Jan", "Feb", "Mar", "April", "May", "Jun"]
    products = ["Product A", "Product B", "Product C"]

    # Generate 5×3 matrix of random grades between 60 and 100 (inclusive)
    salesMat = np.random.randint(1000, 5001, size=(len(months), len(products)))

    # printing the sales matrix
    for i in range(len(months)):
        for j in range(len(products)):
            print(salesMat[i][j], end=" ")
        print()

    # Total monthly sales (sum across products)
    monthlySales = np.sum(salesMat, axis=1)

    # Total sales per product (sum across months)
    productSales = np.sum(salesMat, axis=0)

    maxMonthlySalesIndex = np.argmax(monthlySales)
    maxProductSalesIndex = np.argmax(productSales)

    print("Total monthly sales: ", monthlySales)
    print("Total sales per product: ", productSales)

    print("Max monthly sales: ", months[maxMonthlySalesIndex])
    print("Max sales per product: ", products[maxProductSalesIndex])

# salesDataAnalysis()