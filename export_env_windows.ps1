
$envFilePath = "./.env"

if (Test-Path $envFilePath) {

    Get-Content $envFilePath | ForEach-Object {

        $line = $_ -split "="

        if ($line.Length -eq 2) {

            $key = $line[0].Trim()
            $value = $line[1].Trim()

            # Set the environment variable for the current session
            $env:$key = $value
            Write-Output "Set $key=$value"
        }
    }
} else {
    Write-Output "The .env file was not found at the specified path: $envFilePath"
}
