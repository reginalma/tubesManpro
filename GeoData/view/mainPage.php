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
        <div class="w3-container header">
			<h1 style="margin: 0px; padding-top: 5px; margin-bottom: -37px; padding-top: 10px; padding-left: 20px"><a href="mainPage" class="mainPageLink">GeoData</a></h1>
            <div class="w3-bar" style="height: 50px; padding-right: 20px">
                    
                <button class="w3-bar-item w3-button w3-right buttonPrakira">
                    <h4><a href="prakiraCuaca" id="menuPrakira">Prakira Cuaca</a></h4>
                </button>
    
                <button class="w3-bar-item w3-button w3-right buttonHistory">
                    <div class="w3-dropdown-click hist">
                        <h4 onclick="dropHistory()" >History Cuaca</h4>
                        <div id="Demo" class="w3-dropdown-content w3-bar-block w3-border buttonHistoryContent">
                            <a href="#" class="w3-bar-item w3-button">Albury</a>
                            <a href="#" class="w3-bar-item w3-button">Badgerys Creek</a>
                            <a href="#" class="w3-bar-item w3-button">Cobar</a>
                            <a href="#" class="w3-bar-item w3-button">Coffs Harbour</a>
                            <a href="#" class="w3-bar-item w3-button">Moree</a>
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

        <!-- <div class="w3-light-grey w3-bar">
			<div class="w3-bar-item kota">
                <h2>Albury</h2>
			</div>
			<div class="w3-bar-item status">
                <h2>Cerah</h2>
			</div>
			<div class="w3-bar-item temp">
                <h2>28C</h2>
			</div>
		</div> -->
        <div class="bar">
			<div class="kota">
                <h2>Albury</h2>
			</div>
			<div class="status">
                <h2>Cerah</h2>
			</div>
			<div class="temp">
                <h2>28C</h2>
			</div>
		</div>
        <br>
        <!-- <div class="w3-light-grey w3-bar">
			<div class="w3-bar-item kota">
                <h2>Badgerys Creek</h2>
			</div>
			<div class="w3-bar-item status">
                <h2>Hujan</h2>
			</div>
			<div class="w3-bar-item temp">
                <h2>13C</h2>
			</div>
		</div> -->
        <div class="bar">
			<div class="kota">
                <h2>Badgerys Creek</h2>
			</div>
			<div class="status">
                <h2>Hujan</h2>
			</div>
			<div class="temp">
                <h2>13C</h2>
			</div>
		</div>
        <br>
        <!-- <div class="w3-light-grey w3-bar">
			<div class="w3-bar-item kota">
                <h2>Cobar</h2>
			</div>
			<div class="w3-bar-item status">
                <h2>Cerah</h2>
			</div>
			<div class="w3-bar-item temp">
                <h2>29C</h2>
			</div>
		</div> -->
        <div class="bar">
			<div class="kota">
                <h2>Cobar</h2>
			</div>
			<div class="status">
                <h2>Cerah</h2>
			</div>
			<div class="temp">
                <h2>29C</h2>
			</div>
		</div>
        <br>
        <!-- <div class="w3-light-grey w3-bar">
			<div class="w3-bar-item kota">
                <h2>Coffs Harbour</h2>
			</div>
			<div class="w3-bar-item status">
                <h2>Cerah</h2>
			</div>
			<div class="w3-bar-item temp">
                <h2>27C</h2>
			</div>
		</div> -->
        <div class="bar">
			<div class="kota">
                <h2>Coffs Harbour</h2>
			</div>
			<div class="status">
                <h2>Cerah</h2>
			</div>
			<div class="temp">
                <h2>27C</h2>
			</div>
		</div>
        <br>
        <!-- <div class="w3-light-grey w3-bar">
			<div class="w3-bar-item kota">
                <h2>Moree</h2>
			</div>
			<div class="w3-bar-item status">
                <h2>Hujan</h2>
			</div>
			<div class="w3-bar-item temp">
                <h2>10C</h2>
			</div>
		</div> -->
        <div class="bar">
			<div class="kota">
                <h2>Moree</h2>
			</div>
			<div class="status">
                <h2>Hujan</h2>
			</div>
			<div class="temp">
                <h2>10C</h2>
			</div>
		</div>

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