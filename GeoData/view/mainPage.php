<html>
    <head>
        <title>GeoData</title>
        <link href="https://netdna.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
		<script src="https://netdna.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
		<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.4.1/css/all.css" integrity="sha384-5sAR7xN1Nv6T6+dT2mhtzEpVJvfS3NScPQTrOxhwjIuvcA67KV2R5Jz6kr4abQsz" crossorigin="anonymous">
		<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
		<link rel="stylesheet" href="https://www.w3schools.com/lib/w3-colors-flat.css">
        <link rel='stylesheet' href='https://fonts.googleapis.com/css?family=Roboto'>
		<link rel="stylesheet" href="view/style/style.css">
    </head>
    <body>
        <?php
            $array_kota = array(
                'Albury', 
                'BadgerysCreek', 
                'Cobar', 
                'CoffsHarbour', 
                'Moree',
                'Newcastle', 
                'NorahHead', 
                'NorfolkIsland', 
                'Penrith', 
                'Richmond',
                'Sydney', 
                'SydneyAirport', 
                'WaggaWagga', 
                'Williamtown',
                'Wollongong', 
                'Canberra', 
                'Tuggeranong', 
                'MountGinini', 
                'Ballarat',
                'Bendigo', 
                'Sale', 
                'MelbourneAirport', 
                'Melbourne', 
                'Mildura',
                'Nhil', 
                'Portland', 
                'Watsonia', 
                'Dartmoor', 
                'Brisbane', 
                'Cairns',
                'GoldCoast', 
                'Townsville', 
                'Adelaide', 
                'MountGambier', 
                'Nuriootpa',
                'Woomera', 
                'Albany', 
                'Witchcliffe', 
                'PearceRAAF', 
                'PerthAirport',
                'Perth', 
                'SalmonGums', 
                'Walpole', 
                'Hobart', 
                'Launceston',
                'AliceSprings', 
                'Darwin', 
                'Katherine', 
                'Uluru'
            );
        ?>
        <div class="w3-container header">
            <h1 style="margin: 0px; margin-bottom: -37px; padding-top: 20px; padding-left: 20px">
                <a href="mainPage" class="mainPageLink">
                    GeoData
                </a>
            </h1>
            <div class="w3-bar" style="padding-right: 20px">
                <button class="w3-bar-item w3-button w3-right buttonPrakira">
                    <h4><a href="prakiraCuaca" id="menuPrakira">Prakira Cuaca</a></h4>
                </button>
                <button class="w3-bar-item w3-button w3-right buttonHistory">
                    <div class="w3-dropdown-click hist">
                        <h4 onclick="dropHistory()" >History Cuaca</h4>
                        <div id="Demo" class="w3-dropdown-content w3-bar-block w3-border buttonHistoryContent">
                            <?php
                                foreach($array_kota as $k) {
                                    echo "<a href='historyCuaca?kota=$k' class='w3-bar-item w3-button'>$k</a>";
                                }
                            ?>
                        </div>
                    </div>
                </button>
            </div>
		</div>
        <br>
        <div class="cuaca">
            <h1>Cuaca Hari Ini</h1>
        </div>
        <br>
        <?php
            foreach($array_kota as $k) {
                echo "<div class='bar'>";
                echo     "<div class='kota'>";
                echo         "<h2>$k</h2>";
                echo     "</div>";

                $command_stat = escapeshellcmd("python ..\\GeoData\\model\\WeatherStatus.py $k");
                $stat = shell_exec($command_stat);
                echo     "<div class='status'>";
                echo         "<h2>$stat</h2>";
                echo     "</div>";

                $command_temp = escapeshellcmd("python ..\\GeoData\\model\\WeatherTemp.py $k");
                $temp = shell_exec($command_temp);
                echo     "<div class='temp'>";
                echo         "<h2>$temp&deg;C</h2>";
                echo     "</div>";
		        echo "</div>";
                echo "<br>";
            }
        ?>
        <script>
            function dropHistory() {
                var x = document.getElementById("Demo");
                if (x.className.indexOf("w3-show") == -1) {
                    x.className += " w3-show";
                } else { 
                    x.className = x.className.replace(" w3-show", "");
                }
            }
        </script>
    </body>
</html>