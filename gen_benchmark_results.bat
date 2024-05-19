
:: This scripts runs the ./sgemm binary for all exiting kernels, and logs
:: the outputs to text files in benchmark_results/. Then it calls
:: the plotting script

mkdir benchmark_results

for /l %%x in (10, -1, 0) do (
   echo %%x
   echo.  
   .\build\Release\sgemm.exe %%x | tee "benchmark_results/%%~x_output.txt" || goto :error

   timeout /t 2
)

python3 plot_benchmark_results.py

:error
echo "Are you sure you have build the Release configuration?"
echo "cmake --build . --config Release"
exit /b 1