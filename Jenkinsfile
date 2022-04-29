pipeline {
    agent any
    stages{
        stage ('Build'){
            steps{
                echo 'Building...'
                bat 'python lab.py'
            }
        }
        stage ('Test'){
            steps{
                echo 'Using maven'
            }
        }
    }
}